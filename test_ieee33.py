import pandapower as pp
import pandapower.networks as nw

net = nw.case33bw()  # standard IEEE 33-bus radial distribution test system
print("IEEE 33-bus system is successfully loaded")
print(f"Total buses: {len(net.bus)}")
print(f"Total lines: {len(net.line)}")

pp.runpp(net, numba=False)  # numba=False avoids the speed warning
print("Power flow calculation completed successfully!")
print(net.res_bus[["vm_pu", "va_degree"]].head())
print(f"Minimum voltage: {net.res_bus.vm_pu.min():.4f} p.u.")
