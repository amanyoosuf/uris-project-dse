"""Compare the base case (1.0x load) with commonly reported IEEE 33-bus values.
Reference values are approximate (Baran & Wu 1989); confirm against the source you cite."""
import pandapower as pp
import pandapower.networks as nw

net = nw.case33bw()
pp.runpp(net, numba=False)
vm = net.res_bus.vm_pu
bus = int(vm.idxmin())
rows = [("Total active load (MW)", float(net.load.p_mw.sum()), 3.715, 0.01),
        ("Total reactive load (MVAr)", float(net.load.q_mvar.sum()), 2.300, 0.01),
        ("Minimum voltage (p.u.)", float(vm.min()), 0.9131, 0.002),
        ("Total line losses (kW)", float(net.res_line.pl_mw.sum()) * 1000, 202.7, 5.0)]
print(f"{'Quantity':30s}{'Yours':>10s}{'Reference':>11s}  Result")
for name, mine, ref, tol in rows:
    print(f"{name:30s}{mine:10.4f}{ref:11.4f}  {'PASS' if abs(mine - ref) <= tol else 'CHECK'}")
print(f"Minimum voltage is at bus index {bus} (IEEE bus {bus + 1}); literature reports IEEE bus 18.")
