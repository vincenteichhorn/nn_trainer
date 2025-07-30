import glob
import os
import argparse
from tqdm import tqdm

from nnt.profiling.nvidia_profiler import NvidiaProfiler
from nnt.util.monitor import Monitor

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate total energy of a system.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing energy files.")

    args = parser.parse_args()

    energy_files = glob.glob(os.path.join(args.dir, "**", "energy.csv"), recursive=True)
    print(f"Found {len(energy_files)} energy files.")

    energies = []
    times = []
    for energy_file in Monitor().tqdm(energy_files):
        prof = NvidiaProfiler.from_cache(energy_file)
        prof_energy = prof.get_total_energy()
        energies.append(abs(prof_energy))
        prof_time = prof.get_total_time()
        times.append(abs(prof_time))
    total_energy = sum(energies)

    print(f"Total energy: {total_energy:.2f} J")
    print(f"Total energy: {total_energy / 3600:.2f} Wh")
    print(f"Total energy: {total_energy / 3600 / 1000:.2f} kWh")

    total_time = sum(times)
    print(f"Total time: {total_time:.2f} s")
    print(f"Total time: {total_time / 60:.2f} min")
    print(f"Total time: {total_time / 3600:.2f} h")
    print(f"Total time: {total_time / 3600 / 24:.2f} d")
