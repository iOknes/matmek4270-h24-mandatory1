import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij', sparse=sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx * np.pi
        ky = self.my * np.pi
        return self.c * np.sqrt(kx**2 + ky**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.N, self.mx, self.my = N, mx, my

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        dx = dy = self.h
        return [np.sqrt(dx*dy*np.sum((u - sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0))**2))]

    def apply_bcs(self, Unp1):
        Unp1[0] = 0
        Unp1[-1] = 0
        Unp1[:, -1] = 0
        Unp1[:, 0] = 0
        return Unp1

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """

        # Do an extra run, as recommended on the course page
        Nt += 1

        L = 1
        self.h = L / N
        self.c, self.cfl = c, cfl

        self.create_mesh(N)
        self.initialize(N, mx, my)

        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))

        Unm1[:] = sp.lambdify((x, y, t), self.ue(mx, my))(self.xij, self.yij, 0)
        D = self.D2(N)/self.h**2
        Un[:] = sp.lambdify((x, y, t), self.ue(mx, my))(self.xij, self.yij, self.dt)
        plotdata = {0: Unm1.copy()}
        if store_data == 1:
            plotdata[1] = Un.copy()
        for n in range(1, Nt):
            Unp1[:] = 2*Un - Unm1 + (c*self.dt)**2*(D @ Un + Un @ D.T)
            # Set boundary conditions
            Unp1 = self.apply_bcs(Unp1)
            # Swap solutions
            Unm1[:] = Un
            Un[:] = Unp1
            if n % store_data == 0:
                plotdata[n] = Unm1.copy() # Unm1 is now swapped to Un

        if store_data > 0:
            return plotdata
        if store_data == -1:
            return (self.h, self.l2_error(Un, self.dt * Nt))
        return None

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self, Unp1):
        return Unp1

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol_d = Wave2D()
    r_d, E_d, h_d = sol_d.convergence_rates(cfl=1/np.sqrt(2), mx=2, my=2)
    assert E_d[-1] < 1e-12

    sol_n = Wave2D_Neumann()
    r_n, E_n, h_n = sol_n.convergence_rates(cfl=1/np.sqrt(2), mx=2, my=2)
    assert E_n[-1] < 1e-12

def main():
    import matplotlib.animation as animation
    from matplotlib import cm

    sol = Wave2D_Neumann()
    data = sol(16, 60, store_data=1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in data.items():
        #frame = ax.plot_wireframe(sol.xij, sol.yij, val, rstride=2, cstride=2);
        frame = ax.plot_surface(sol.xij, sol.yij, val, vmin=-0.5*data[0].max(),
                                vmax=data[0].max(), cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                    repeat_delay=1000)
    ani.save('wavemovie2d.gif', writer='pillow', fps=12)

if __name__ == "__main__":
    main()
