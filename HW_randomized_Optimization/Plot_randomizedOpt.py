import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_search():
    files_to_plot = ["genetic_alg_search.xlsx", "mimic_search.xlsx", "simulated_annealing_search.xlsx"]
    for file in files_to_plot:
        algo_name = file[:-5]
        print(algo_name)
        df = pd.read_excel(file)

        for problem, problem_df in df.groupby(['problem']):
            if file == "genetic_alg_search.xlsx":
                for pop_size, pop_size_df in problem_df.groupby(['pop_size']):
                    pop_size_df = pop_size_df.sort_values('mutation_prob')
                    if 195 < pop_size < 210:
                        plot(pop_size_df, "mutation_prob", problem, algo_name, key="pop_size", value =pop_size)

                for mutation_prob, mutation_prob_df in problem_df.groupby(['mutation_prob']):
                    mutation_prob_df = mutation_prob_df.sort_values('pop_size')
                    if mutation_prob == 0.1:
                        plot(mutation_prob_df, "pop_size", problem, algo_name, key="mutation_prob", value=int(mutation_prob*100))

            elif file == "simulated_annealing_search.xlsx":
                plot(problem_df, "decay_name", problem, algo_name, key="decay_name", value="4_value")
            #
            else:
                problem_df = problem_df.sort_values('keep_pct')
                plot(problem_df, "keep_pct", problem, algo_name, key="keep_pct", value="plot")


def plot(df, attribute, problem, algo_name, key, value):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{problem} problem {algo_name} {attribute} plot", fontsize=20)
    axes[0].grid()
    axes[0].set_title(f"fitness score for {attribute}")
    # axes[0].set_ylim(0.5, 1)
    axes[0].set_xlabel(attribute)
    axes[0].set_ylabel("fitness score")
    # axes[0].fill_between(x_value, score_mean - score_std,
    #                      score_mean + score_std, alpha=0.1, color="g")
    axes[0].plot(df[attribute], df["score"], 'o-', color="g", label=f"Score with respect to {attribute}")
    axes[0].legend(loc="best")

    axes[1].grid()
    axes[1].plot(df[attribute], df["time"], 'o-')
    # axes[1].fill_between(x_value, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel(attribute)
    axes[1].set_ylabel("run times in seconds")
    axes[1].set_title("Scalability of the model")
    print(os.path.join(r"C:\Users\Lijun\Box\Gatech\CS7641_ML\HW_randomized_Optimization\Graph\Search_plot", f"{algo_name}#{problem}#Param_{attribute} {key}={value}"))
    fig.savefig(os.path.join(r"C:\Users\Lijun\Box\Gatech\CS7641_ML\HW_randomized_Optimization\Graph\Search_plot", f"{problem}#{algo_name}#Param_{attribute} {key}={value}"))
    fig.clf()

if __name__ == "__main__":
    plot_search()