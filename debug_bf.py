from main import compile_all, single_run


if __name__ == "__main__":
    compile_all()
    single_run(
        config_filename="experiment_bf.json",
        snr_db=7.0,
    )