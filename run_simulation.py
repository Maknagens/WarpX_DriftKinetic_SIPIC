"""Drift kinetic SIPIC model"""

import argparse
import os
import shutil
import time

import numpy as np
from scipy.interpolate import griddata
from mpi4py import MPI as mpi
os.environ[ 'MPLCONFIGDIR' ] = './tmp/'
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from pywarpx import callbacks, fields, particle_containers, picmi, libwarpx, warpx, algo


constants = picmi.constants

comm = mpi.COMM_WORLD
simulation = picmi.Simulation(
    verbose=0,
    particle_shape=1,
    warpx_current_deposition_algo='direct',
    warpx_use_filter=False,
    warpx_load_balance_intervals=1000,        # re-balance every N steps
    warpx_load_balance_efficiency_ratio_threshold=1.1,  # only rebalance if imbalance > 10%
    warpx_load_balance_costs_update='Heuristic',  # or 'Timers'
    warpx_particle_pusher_algo = "DriftKinetic",
)
#simulation.amr_restart = "./checkpoints/Checkpoint_01450000"
# make a shorthand for simulation.extension since we use it a lot
sim_ext = simulation.extension


class DriftKineticSolver(picmi.ElectrostaticSolver):
    right_wall_charge = 0.

    def __init__(self, grid, **kwargs):

        super(DriftKineticSolver, self).__init__(
            grid=grid, method=kwargs.pop('method', 'Multigrid'),
            required_precision=1e-6, warpx_self_fields_verbosity = 0
        )

    def solver_initialize_inputs(self):

        super().solver_initialize_inputs()
        warpx.do_electrostatic = "LabFrameDriftKinetic"
        warpx.effective_potential_factor = 4
        warpx.drift_kinetic_reference_B  = 0.1   # B at z_lo
        
        
        self.nppc = MirrorDriftKinetic.nppc
        self.L = MirrorDriftKinetic.L
        self.R = MirrorDriftKinetic.R
        self.nz = MirrorDriftKinetic.nz
        self.dz = self.L / self.nz

        simulation.applied_fields = [
            picmi.AnalyticInitialField(
                Bx_expression=0.0,
                By_expression=0.0,
                Bz_expression=f"(1./(1.+({self.R}-1.)*((z-{self.L/2.})/{self.L}*2.)**2))",
                warpx_do_initial_div_cleaning = 0, # Otherwise the field is wiped out
            )
        ]
        
class MirrorDriftKinetic(object):

    #######################################################################
    # Begin physical parameters                                           #
    #######################################################################

    L = 1.2   # m

    m_ion = constants.m_p
    nz = 1024

    nppc = 10 # Amount of injected particle per time step
    A0 = 1. # Reference tube area
    flux = 10.0 # A/m^2
    n0 = 4e15 # Target density to get Debye length

    R = 10.
    K = 10.

    #######################################################################
    # End global user parameters and user input                           #
    #######################################################################

    def __init__(self, Ti, Te):
        """
        Arguments:
            Ti (float): Ion temperature in eV.
        """
        self.Ti = Ti
        self.Te = Te
        self.c_s = np.sqrt((self.Te + self.Ti/2.) * constants.q_e / self.m_ion)
        self.debye = 7430. * np.sqrt(Te / self.n0)
        self.plasmfreq = (self.n0 * constants.q_e**2 / constants.m_e / constants.ep0)
        self.vTe = np.sqrt(2. * self.Te * constants.q_e / constants.m_e)
        self.vTi = np.sqrt(2. * self.Ti * constants.q_e / self.m_ion)

        self.ve_max = 4. * self.vTe
        self.vi_max = 4. * self.vTi
        #self.flux = self.n0 * constants.q_e * self.c_s

        self.dz = self.L / self.nz
        self.dt = 2.5e-11

        self.B0 = self.get_Bz(0.0)

        # calculate rough crossing time steps for ions
        self.total_steps = 5000000 # Maximum steps
        self.diag_steps = 1000 # Diagnostic output step

        if comm.rank == 0:
            print("Starting simulation with parameters:")
            print(f"    T_e = {self.Te:.3f} eV")
            print(f"    T_i = {self.Ti:.3f} eV")
            print(f"    c_s = {self.c_s:.3e} m/s")
            print(f"    M/m = {self.m_ion/constants.m_e:.0f}")
            print(f"    n0 = {self.n0:.1e} m^-3")
            print(f"    flux = {self.flux:.1f} A")
            print(f"    Total steps = {self.total_steps}")
            print("")


        #######################################################################
        # Set geometry, boundary conditions and timestep                      #
        #######################################################################
        u_e_th = np.sqrt(self.Te/9.1e-31 * 1.6e-19) / 3e8
        u_i_th = np.sqrt(self.Ti/self.m_ion * 1.6e-19) / 3e8 
        warpx_boundary_u_th = { "electrons": u_e_th, "ions": u_i_th}
        
        self.grid = picmi.Cartesian1DGrid(
            number_of_cells=[self.nz],
            warpx_max_grid_size=8,
            lower_bound=[0.0],
            upper_bound=[self.L],
            lower_boundary_conditions=['dirichlet'],
            upper_boundary_conditions=['neumann'], # neumann -> floating, dirichlet -> dirichelt
            lower_boundary_conditions_particles=['thermal'],
            upper_boundary_conditions_particles=['absorbing'],
            warpx_boundary_u_th = warpx_boundary_u_th,
            warpx_potential_hi_z = -4.2 * self.Te,
        )
        simulation.time_step_size = self.dt
        simulation.max_steps = self.total_steps
        # simulation.load_balance_intervals = self.total_steps // 1000

        #######################################################################
        # Field solver and external field                                     #
        #######################################################################

        simulation.solver = DriftKineticSolver(grid=self.grid, Te=self.Te)

        #######################################################################
        # Particle types setup                                                #
        #######################################################################
        self.electrons = picmi.Species(
            particle_type='electron', name='electrons',
            #warpx_save_particles_at_zlo=True,
            warpx_save_particles_at_zhi=True, # important for floating wall in the solver, 
                                              # the solver cleans this by itself,
                                              # the solver writes charge difference into the drift_kinetic_wall_charge.txt
                                              # to not modify checkpoint file
        )

        self.ions = picmi.Species(
            particle_type='H', name='ions', charge_state=1,
            mass=self.m_ion,
            #warpx_save_particles_at_zlo=True,
            warpx_save_particles_at_zhi=True, # important for floating wall in the solver, 
                                              # the solver cleans this by itself,
                                              # the solver writes charge difference into the drift_kinetic_wall_charge.txt
                                              # to not modify checkpoint file
        )

        layout = picmi.GriddedLayout(
            n_macroparticle_per_cell=[200], grid=self.grid
        )

        simulation.add_species(self.ions, layout=layout, initialize_self_field=True)
        simulation.add_species(self.electrons, layout=layout, initialize_self_field=True)
        #######################################################################
        # Checkpoint setup                                                #
        #######################################################################
        diagnostic = picmi.Checkpoint(

            period=f'{50000}::{50000}',
            name="Checkpoint_",
            write_dir="./checkpoints",
            warpx_file_min_digits=8
        )
        simulation.add_diagnostic(diagnostic)
        #######################################################################
        # Particle injection                                                  #
        #######################################################################

        self.nppc = self.nppc

        self.weight = (
            self.flux / constants.q_e * self.A0 * self.dt
            / (self.nppc)
        )

        self.sigma_i = np.sqrt(self.Ti * constants.q_e / self.m_ion)
        self.sigma_e = np.sqrt(self.Te * constants.q_e / constants.m_e)
        callbacks.installparticleinjection(self.particle_injection)
        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################

        callbacks.installafterinit(self._create_diags_dir)
        callbacks.installafterstep(self.text_diag)
        callbacks.installafterstep(self.phi_diag)
        callbacks.installafterstep(self.E_diag)
        callbacks.installafterstep(self.rho_diag)
        callbacks.installafterstep(self.ion_diags)
        callbacks.installafterstep(self.electron_diags)
        # #callbacks.installafterstep(self.current_diag)
        callbacks.installafterstep(self.phase_space_diag)

        # the current diagnostic will fill an array of length at most N
        # if more simulation steps will be taken than that, steps will be
        # averaged together
        timeseries_len = 800
        if self.total_steps <= timeseries_len:
            self.current_steps = 1
            timeseries_len = self.total_steps
        else:
            while(timeseries_len >= 2):
                if self.total_steps % timeseries_len == 0:
                    break
                timeseries_len-=1
            self.current_steps = self.total_steps // timeseries_len

        if comm.rank == 0:
            self.currents = np.zeros((2, timeseries_len))

        # the phase space plots we are interested in plots z vs vz
        # we will sample the particles every couple of steps
        self.z_bins = np.linspace(0, 1, 200)
        self.vz_bins = np.linspace(-5, 5, 200)
        self.vperp_bins = np.linspace(0, 5, 200)
        self.phase_space = np.zeros((len(self.z_bins)-1, len(self.vz_bins)-1)).T
        self.phase_space_perp = np.zeros((len(self.z_bins)-1, len(self.vperp_bins)-1)).T
        self.phase_space_vels = np.zeros((len(self.vz_bins)-1, len(self.vperp_bins)-1)).T

        self.ps_snapshots = 10
        while self.ps_snapshots < self.diag_steps:
            if self.diag_steps % self.ps_snapshots != 0:
                self.ps_snapshots += 1
            else:
                break

        if comm.rank == 0:
            print("    gathering phase space every "
                  f"{self.diag_steps/self.ps_snapshots} steps\n"
            )
        if self.ps_snapshots > self.diag_steps / 4:
            if comm.rank == 0:
                print("Too many snapshots for the phase space diag")
            exit()

        #######################################################################
        # Initialize run and print diagnostic info                            #
        #######################################################################

        simulation.initialize_inputs()
        simulation.initialize_warpx()
        global sim_ext
        sim_ext = simulation.extension.warpx
        # get a reference to the ion particle container
        self.ion_cont = particle_containers.ParticleContainerWrapper(
            self.ions.name
        )
        self.electron_cont = particle_containers.ParticleContainerWrapper(
            self.electrons.name
        )
        # get a reference to the particle boundary buffer
        self.boundary_buffer = particle_containers.ParticleBoundaryBufferWrapper()

    def particle_injection(self):

        if (comm.rank == 0):
            z_vals = (1. - np.random.rand(self.nppc)**(0.25)) * self.L / 10.
            vx_vals = np.random.normal(0, self.sigma_i, self.nppc)
            vy_vals = np.random.normal(0, self.sigma_i, self.nppc)
            vz_vals = self.sigma_i * np.sqrt(-2. * np.log(1.0 - np.random.rand(self.nppc)))
            self.ion_cont.add_particles(
                z=z_vals,
                ux=np.sqrt(vx_vals**2 + vy_vals**2),
                uy=0.0,
                uz=vz_vals,
                w=self.weight
            )

            vx_vals = np.random.normal(0, self.sigma_e, self.nppc)
            vy_vals = np.random.normal(0, self.sigma_e, self.nppc)
            vz_vals = self.sigma_e * np.sqrt(-2. * np.log(1.0 - np.random.rand(self.nppc)))

            self.electron_cont.add_particles(
                z=z_vals,
                ux=np.sqrt(vx_vals**2 + vy_vals**2),
                uy=0.0,
                uz=vz_vals,
                w=self.weight
            )
        else:
            self.ion_cont.add_particles(
                z=None,
                ux=None,
                uy=None,
                uz=None,
                w=None,

            )
            self.electron_cont.add_particles(
                z=None,
                ux=None,
                uy=None,
                uz=None,
                w=None,
            )
          
    def get_Bz(self, z):
        return(1./(1. + (self.R-1.) * ((z - self.L/2.)/self.L * 2.)**2))
              
    def get_dBdz(self, z):
        return(- (4.*self.L**2 * (self.R - 1.) * (2.*z - self.L)) /
               (self.L**2*self.R - 4. * self.L * (self.R - 1.) * z + 4. * (self.R - 1.) * z**2)**2
          )

    def _create_diags_dir(self):
        if comm.rank == 0:
            if os.path.exists('diags'):
                shutil.rmtree('diags')
            os.mkdir('diags')
            os.mkdir('diags/phi')
            os.mkdir('diags/E')
            os.mkdir('diags/rho')
            os.mkdir('diags/flux')
            os.mkdir('diags/phase_space_i')
            os.mkdir('diags/phase_space_e')
            os.mkdir('diags/vzi')
            os.mkdir('diags/ni')
            os.mkdir('diags/vze')
            os.mkdir('diags/ne')

            fig, ax = plt.subplots(figsize=(4, 3))
            z_grid = np.linspace(0, self.L, 500)
            ax.plot(z_grid / self.L, self.get_Bz(z_grid))
            ax.grid()
            ax.set_xlabel('z/L')
            ax.set_ylabel('B(z) [T]')
            plt.tight_layout()
            plt.savefig("diags/B_profile.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(4, 3))
            z_grid = np.linspace(0, self.L, 500)
            ax.plot(z_grid / self.L, self.get_dBdz(z_grid))
            ax.grid()
            ax.set_xlabel('z/L')
            ax.set_ylabel('dB(z)/dz [T/m]')
            plt.tight_layout()
            plt.savefig("diags/dBdz_profile.png")
            plt.close()
    def text_diag(self):
        """Diagnostic function to print out timing data and particle numbers."""
        step = sim_ext.getistep(0)
        if step % self.diag_steps != 0:
            return
        #print("ions amount on processor: " + str(self.ion_cont.get_particle_count(local=True)) +
        #      ", electrons amount on processor: " + str(self.electron_cont.get_particle_count(local=True)) +
        #      ", processor: " + str(comm.rank) + ", iteration_num = " + str(sim_ext.getistep(0)), flush=True)
        wall_time = time.time() - self.prev_time
        steps = step - self.prev_step
        step_rate = steps / wall_time
        simtime = self.dt * step * 1e6
        status_dict = {
            'step': step,
            'nplive ions': self.ion_cont.get_particle_count(local=False),
            'nplive electrons': self.electron_cont.get_particle_count(local=False),
            'wall_time': wall_time,
            'step_rate': step_rate,
            "simtime": simtime,
            'iproc': None
        }

        diag_string = (
            "Step #{step:6d}; "
            "{nplive ions} ions; "
            "{nplive electrons} electrons; "
            "{wall_time:6.1f} s wall time; "
            "{step_rate:4.2f} steps/s; "
            "{simtime:4.2f} total time"
        )
        if comm.rank == 0:
            print(diag_string.format(**status_dict), flush=True)

        self.prev_time = time.time()
        self.prev_step = step
    def phi_diag(self):
    
        step = sim_ext.getistep(0)
        if step % self.diag_steps != 0:
            return
            
        data = fields.PhiFPWrapper()[...] / self.Te
        if comm.rank != 0:
            return
        
        np.save(f"diags/phi/phi_{step:08d}.npy", data)

        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.xlabel('x/L')
        plt.ylabel("e$\phi/T_e$")
        plt.grid()
        plt.savefig(f"diags/phi/phi_{step:06d}.png")
        plt.close()
    def E_diag(self):
        step = sim_ext.getistep(0)
        if step % self.diag_steps != 0:
            return
        data = fields.EzWrapper()[...]
        if comm.rank != 0:
            return

        np.save(f"diags/E/E_{step:08d}.npy", data)

        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.xlabel('x/L')
        plt.ylabel("E, V/m$")
        plt.grid()
        plt.savefig(f"diags/E/E_{step:06d}.png")
        plt.close()
    def rho_diag(self):

        step = sim_ext.getistep(0)
        if step % self.diag_steps != 0:
            return
        data = fields.RhoFPWrapper()[...] / constants.q_e #fields.RhoFPWrapper()[...] / constants.q_e# / self.n0
        if comm.rank != 0:
            return
   
        np.save(f"diags/rho/rho_{step:08d}.npy", data)
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.xlabel('x/L')
        plt.ylabel("$n_{ion}/n_0$")
        plt.grid()
        #plt.yscale('log')
        plt.savefig(f"diags/rho/rho_{step:06d}.png")
        plt.close()
    def ion_diags(self):
        step = sim_ext.getistep(0)
        if step % self.diag_steps != 0:
            return
        data = fields.EzWrapper()[...]

        dat_size = len(data)
        grid_spacing = 1. / dat_size

        n = self.ion_cont.get_particle_count(local=True)

        if n > 0:
            z = np.concatenate(self.ion_cont.zp) / self.L
            uz = np.concatenate(self.ion_cont.uzp) / self.c_s
            w = np.concatenate(self.ion_cont.wp)
            sum_w = np.zeros_like(data)
            sum_uz = np.zeros_like(data)
            cnt = np.zeros_like(data)

            for i in range(len(z)):
                z_local = z[i]
                w_local = w[i] * self.get_Bz(z[i]) / self.get_Bz(0)
                uz_local = uz[i]

                idx = int(z_local / grid_spacing)
                sum_w[idx] += w_local
                sum_uz[idx] += uz_local
                cnt[idx] += 1
        else:
            sum_w = np.zeros_like(data)
            sum_uz = np.zeros_like(data)
            cnt = np.zeros_like(data)

        sum_w_global = comm.allreduce(sum_w, op=mpi.SUM)
        sum_uz_global = comm.allreduce(sum_uz, op=mpi.SUM)
        cnt_global = comm.allreduce(cnt, op=mpi.SUM)

        if comm.rank != 0:
            return
        #print(str(cnt) + " " + str(cnt_global) + " " + str(n))
        sum_w_global /= self.dz
        sum_uz_global /= (cnt_global + 1)

        np.save(f"diags/vzi/vzi_{step:08d}.npy", sum_uz_global)
        plt.plot(range(0, dat_size), sum_uz_global)
        plt.xlabel('x/L')
        plt.ylabel("$M$")
        plt.grid()
        # plt.yscale('log')
        plt.savefig(f"diags/vzi/vzi_{step:06d}.png")
        plt.close()

        np.save(f"diags/ni/ni_{step:08d}.npy", sum_w_global)
        plt.plot(range(0, dat_size), sum_w_global)
        plt.xlabel('x/L')
        plt.ylabel("$n_e, m^{-3}$")
        plt.grid()
        plt.yscale('log')
        plt.savefig(f"diags/ni/ni_{step:06d}.png")
        plt.close()
    def electron_diags(self):
        step = sim_ext.getistep(0)
        if step % self.diag_steps != 0:
            return

        data = fields.EzWrapper()[...]

        dat_size = len(data)
        grid_spacing = 1. / dat_size
        n = self.electron_cont.get_particle_count(local=True)
        if n > 0:
            z = np.concatenate(self.electron_cont.zp) / self.L
            uz = np.concatenate(self.electron_cont.uzp) / self.sigma_e / np.sqrt(2)
            w = np.concatenate(self.electron_cont.wp)

            sum_w = np.zeros_like(data)
            sum_uz = np.zeros_like(data)
            cnt = np.zeros_like(data)

            for i in range(len(z)):
                z_local = z[i]
                w_local = w[i] * self.get_Bz(z[i]) / self.get_Bz(0)
                uz_local = uz[i]

                idx = int(z_local / grid_spacing)
                sum_w[idx] += w_local
                sum_uz[idx] += uz_local
                cnt[idx] += 1
        else:
            sum_w = np.zeros_like(data)
            sum_uz = np.zeros_like(data)
            cnt = np.zeros_like(data)

        sum_w_global = comm.allreduce(sum_w, op=mpi.SUM)
        sum_uz_global = comm.allreduce(sum_uz, op=mpi.SUM)
        cnt_global = comm.allreduce(cnt, op=mpi.SUM)
        if comm.rank != 0:
            return

        sum_w_global /= self.dz
        sum_uz_global /= (cnt_global + 1)

        np.save(f"diags/vze/vze_{step:08d}.npy", sum_uz_global)
        plt.plot(range(0, dat_size), sum_uz_global)
        plt.xlabel('x/L')
        plt.ylabel("$M$")
        plt.grid()
        # plt.yscale('log')
        plt.savefig(f"diags/vze/vze_{step:06d}.png")
        plt.close()

        np.save(f"diags/ne/ne_{step:08d}.npy", sum_w_global)
        plt.plot(range(0, dat_size), sum_w_global)
        plt.xlabel('x/L')
        plt.ylabel("$n_e, m^{-3}$")
        plt.grid()
        plt.yscale('log')
        plt.savefig(f"diags/ne/ne_{step:06d}.png")
        plt.close()

    def current_diag(self):

        step = sim_ext.getistep(0)
        idx = step // self.current_steps - 1
        num = step % self.current_steps

        if num == 0:
            for ii, side in enumerate(['z_lo', 'z_hi']):
                scraped_number = self.boundary_buffer.get_particle_boundary_buffer_size(
                    self.ions.name, side, local=True
                )
                if not scraped_number:
                    my_current = 0.0
                else:
                    part_container = self.boundary_buffer.particle_buffer.get_particle_container(
                        self.ions.name, self.boundary_buffer._get_boundary_number(side)
                    )
                    scraped_weights = np.zeros(scraped_number)
                    idx = 0
                    w_idx = self.ion_cont.particle_container.get_comp_index("w")
                    for pti in libwarpx.libwarpx_so.BoundaryBufferParIter(part_container, 0):
                        soa = pti.soa()
                        w = np.array(soa.GetRealData(w_idx), copy=False)
                        num = len(w)
                        scraped_weights[idx:idx+num] = w

                    my_current = np.sum(scraped_weights) * self.ions.q / self.dt

                total_current = comm.allreduce(my_current, op=mpi.SUM)

                if comm.rank == 0:
                    self.currents[ii, idx] = total_current / self.current_steps

            self.boundary_buffer.clear_buffer()

        # output data if this is a diagnostic step
        if step % self.diag_steps != 0 or comm.rank != 0:
            return

        np.save(f"diags/flux/currents_{step:08d}.npy", self.currents[:,:idx+1])

        times = (np.arange(idx + 1) + 1.0) * self.dt * self.current_steps * 1e6
        plt.plot(times, self.currents[0,:idx+1], label="left current")
        plt.plot(times, self.currents[1,:idx+1], label="right current")
        plt.plot(times, np.ones_like(times)* self.flux * self.A0, 'k--')
        plt.xlabel('Time ($\mu$s)')
        plt.ylabel("Current (A)")
        plt.grid()
        plt.legend()
        plt.savefig(f"diags/flux/current_{step:06d}.png")
        plt.close()

    def phase_space_diag(self):
        step = sim_ext.getistep(0)
###################################### IONS ###################################
        if (step % self.diag_steps) == 0:

            n = self.ion_cont.get_particle_count(local=True)
            if n > 0:
                z = np.concatenate(self.ion_cont.zp) / self.L
                uz = np.concatenate(self.ion_cont.uzp) / self.c_s
                u_perp = np.sqrt(
                    np.concatenate(self.ion_cont.uxp)**2
                    # + np.concatenate(sim_ext.get_particle_uy('ions'))**2
                ) / self.c_s
                w = np.concatenate(self.ion_cont.wp) * self.get_Bz(z) / self.get_Bz(0)

                my_H, xedges, yedges = np.histogram2d(
                    z, uz, bins=[self.z_bins, self.vz_bins], weights=w
                )
                my_H_perp, xedges_perp, yedges_perp = np.histogram2d(
                    z, u_perp, bins=[self.z_bins, self.vperp_bins], weights=w
                )

                idx = np.where(z < 0.1)
                my_H_vels, xedges_vels, yedges_vels = np.histogram2d(
                    uz[idx], u_perp[idx], bins=[self.vz_bins, self.vperp_bins], weights=1/w[idx]
                )
            else:
                my_H = np.zeros_like(self.phase_space).T
                my_H_perp = np.zeros_like(self.phase_space_perp).T
                my_H_vels = np.zeros_like(self.phase_space_vels).T

            H = comm.allreduce(my_H.T, op=mpi.SUM)
            H_perp = comm.allreduce(my_H_perp.T, op=mpi.SUM)
            H_vels = comm.allreduce(my_H_vels.T, op=mpi.SUM)

            if comm.rank == 0:
                np.save(f"diags/phase_space_i/phasespace_parallel_{step:08d}.npy", H)
                np.save(f"diags/phase_space_i/phasespace_perp_{step:08d}.npy", H_perp)
                np.save(f"diags/phase_space_i/phasespace_vels_{step:08d}.npy", H_vels)

                plt.imshow(H + np.min(H[np.nonzero(H)]), cmap='jet', extent = (0.,1., -5, 5), aspect='auto', norm=colors.LogNorm(), origin="lower")#, ax=ax)
                plt.xlabel('z/L')
                plt.ylabel('$v_\parallel/c_s$')
                plt.savefig(f"diags/phase_space_i/phasespace_parallel_{step:06d}.png")
                plt.close()

                plt.imshow(H_perp + np.min(H_perp[np.nonzero(H_perp)]), cmap='jet', extent = (0.,1., 0., 5), aspect='auto', norm=colors.LogNorm(), origin="lower")#, ax=ax)
                plt.xlabel('z/L')
                plt.ylabel('$v_\perp/c_s$')
                plt.savefig(f"diags/phase_space_i/phasespace_perp_{step:06d}.png")
                plt.close()

                plt.imshow(H_vels + np.min(H_vels[np.nonzero(H_vels)]), cmap='jet', extent = (-5.,5., 0, 5), aspect='auto', norm=colors.LogNorm(), origin="lower")#, ax=ax)
                plt.xlabel('$v_\parallel/c_s$')
                plt.ylabel('$v_\perp/c_s$')
                plt.savefig(f"diags/phase_space_i/phasespace_vels_{step:06d}.png")
                plt.close()

    ###################################### ELECTRONS ###################################
        if (step % self.diag_steps) == 0:

            n = self.electron_cont.get_particle_count(local=True)
            if n > 0:
                z = np.concatenate(self.electron_cont.zp) / self.L
                uz = np.concatenate(self.electron_cont.uzp) / self.vTe
                u_perp = np.sqrt(
                    np.concatenate(self.electron_cont.uxp) ** 2
                    # + np.concatenate(sim_ext.get_particle_uy('ions'))**2
                ) / self.vTe
                w = np.concatenate(self.electron_cont.wp) * self.get_Bz(z) / self.get_Bz(0)

                my_H, xedges, yedges = np.histogram2d(
                    z, uz, bins=[self.z_bins, self.vz_bins], weights=w
                )
                my_H_perp, xedges_perp, yedges_perp = np.histogram2d(
                    z, u_perp, bins=[self.z_bins, self.vperp_bins], weights=w
                )

                idx = np.where(z < 0.1)
                my_H_vels, xedges_vels, yedges_vels = np.histogram2d(
                    uz[idx], u_perp[idx], bins=[self.vz_bins, self.vperp_bins], weights=1/w[idx]
                )
            else:
                my_H = np.zeros_like(self.phase_space).T
                my_H_perp = np.zeros_like(self.phase_space_perp).T
                my_H_vels = np.zeros_like(self.phase_space_vels).T

            H = comm.allreduce(my_H.T, op=mpi.SUM)
            H_perp = comm.allreduce(my_H_perp.T, op=mpi.SUM)
            H_vels = comm.allreduce(my_H_vels.T, op=mpi.SUM)


            if comm.rank == 0:
                np.save(f"diags/phase_space_e/phasespace_parallel_{step:08d}.npy", H)
                np.save(f"diags/phase_space_e/phasespace_perp_{step:08d}.npy", H_perp)
                np.save(f"diags/phase_space_e/phasespace_vels_{step:08d}.npy", H_vels)

                plt.imshow(H + np.min(H[np.nonzero(H)]), cmap='jet', extent=(0., 1., -5, 5), aspect='auto', norm=colors.LogNorm(),
                           origin="lower")  # , ax=ax)
                plt.xlabel('z/L')
                plt.ylabel('$v_\parallel/c_s$')
                plt.savefig(f"diags/phase_space_e/phasespace_parallel_{step:06d}.png")
                plt.close()

                plt.imshow(H_perp + np.min(H_perp[np.nonzero(H_perp)]), cmap='jet', extent=(0., 1., 0., 5), aspect='auto',
                           norm=colors.LogNorm(), origin="lower")  # , ax=ax)
                plt.xlabel('z/L')
                plt.ylabel('$v_\perp/c_s$')
                plt.savefig(f"diags/phase_space_e/phasespace_perp_{step:06d}.png")
                plt.close()

                plt.imshow(H_vels + np.min(H_vels[np.nonzero(H_vels)]), cmap='jet', extent=(-5., 5., 0, 5), aspect='auto',
                           norm=colors.LogNorm(), origin="lower")  # , ax=ax)
                plt.xlabel('$v_\parallel/c_s$')
                plt.ylabel('$v_\perp/c_s$')
                plt.savefig(f"diags/phase_space_e/phasespace_vels_{step:06d}.png")
                plt.close()
    def run_sim(self):

        self.prev_time = time.time()
        self.start_time = self.prev_time
        self.prev_step = 0

        simulation.step()


# Create the parser
parser = argparse.ArgumentParser()

# Add an argument
parser.add_argument('--Ti', type=float, required=True)
parser.add_argument('--Te', type=float, required=True)
# Parse the argument
args = parser.parse_args()

my_sim = MirrorDriftKinetic(args.Ti, args.Te)
my_sim.run_sim()
