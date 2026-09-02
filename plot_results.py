"""Run the live results dashboard independently from a simulation job."""

import argparse
import dataclasses
import multiprocessing as mp
import signal

from ldpc_experiment import LdpcExperimentSettings
from simulator_awgn_python.live_plot import PlotServer
from simulator_awgn_python.postprocessing import PostProcessing
from simulator_awgn_python.settings import Settings


def run_plot_server(title, visualization, postprocessing):
    """Run one blocking Dash server in a child process."""
    PlotServer(title, visualization, postprocessing).run()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Serve plots from saved simulation results without running the simulation.'
        )
    )
    parser.add_argument(
        '-c', '--config', required=True,
        help='Experiment JSON file used by the simulation'
    )
    parser.add_argument(
        '--host',
        help='Address to listen on (overrides visualization.ip_address)'
    )
    parser.add_argument(
        '--port', type=int,
        help='First port to use (overrides visualization.start_port)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = Settings(args.config, LdpcExperimentSettings)
    processes = []
    plot_index = 0

    def stop_server(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_server)

    try:
        while settings.remaining():
            visualization, experiment = settings.next_experiment(
                print_exp=False,
                print_vis=False,
            )
            visualization = dataclasses.replace(
                visualization,
                ip_address=args.host or visualization.ip_address,
                start_port=(
                    args.port + plot_index
                    if args.port is not None
                    else visualization.start_port
                ),
            )
            postprocessing = PostProcessing(
                experiment.filename,
                experiment.modulation,
                settings.postproc,
            )
            process = mp.Process(
                target=run_plot_server,
                args=(experiment.title, visualization, postprocessing),
            )
            process.start()
            processes.append(process)

            print(f'Plot data: {experiment.filename}')
            print(f'Plot server: {visualization.url}')
            plot_index += 1

        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print('\nStopping plot server...')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()


if __name__ == '__main__':
    main()
