import main_default as baseline
from data_io import load_concrete_data


def portable_load_data():
    return load_concrete_data(baseline)


baseline.load_data = portable_load_data


if __name__ == "__main__":
    baseline.train_run(
        batch_size=16,
        lr=0.0003,
        epochs=200,
        hidden1_size=32,
        hidden2_size=32,
        dropout=0.2,
    )
    baseline.test_run(
        batch_size=16,
        lr=0.0003,
        epochs=200,
        hidden1_size=32,
        hidden2_size=32,
        dropout=0.2,
    )
