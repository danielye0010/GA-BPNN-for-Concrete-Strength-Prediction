import main_ga as ga_model
from data_io import load_concrete_data


def portable_load_data():
    return load_concrete_data(ga_model)


ga_model.load_data = portable_load_data


if __name__ == "__main__":
    best_parameters = ga_model.ga_optimization()
    ga_model.train_run(best_parameters)
    ga_model.test_run(best_parameters)
