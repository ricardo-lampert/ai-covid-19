import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BAR_WIDTH = 0.25
WIDTH = 10
HIGH = 10
Y_MIN = 0
Y_MAX = 1


def plot_multiple_lines(
    x, result, labels, path="graphic", save=True, title="", axis_label=None
):
    plt.ylim(Y_MIN, Y_MAX)
    x_steps = [step for step in list(range(max(x)+1)) if step % 10 == 0]
    y_steps = np.linspace(0,1,11)
    plt.xticks(x_steps)
    plt.yticks(y_steps)
    for y, label in zip(result, labels):
        # plot lines

        plt.plot(x, y, label=label, linestyle="-")
    axis_label = ("X", "Y") if not axis_label else axis_label
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.title(title)
    # plt.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.05),
    #     fancybox=True,
    #     shadow=False,
    #     ncol=3,
    # )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save:
        plt.tight_layout()
        plt.savefig((path + ".jpg"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_grouped_bars(
    data,
    labels,
    path="graph",
    title="graph",
    colors=None,
    save=True,
    groups=None,
    width=WIDTH,
    high=HIGH,
):
    plt.figure(figsize=(width, high))

    bars = [bar for bar in data]
    x_positions = [np.arange(len(bars[0]))]

    for i in range(1, len(bars)):
        x_positions.append([x + BAR_WIDTH for x in x_positions[i - 1]])

    for i in range(len(bars)):
        plt.bar(
            x_positions[i],
            bars[i],
            color=colors[i] if colors else None,
            width=BAR_WIDTH,
            edgecolor="white",
            label=labels[i],
        )
    plt.xlabel(title, fontweight="bold")
    if len(bars[0]) == 1:
        plt.tick_params(bottom=False, labelbottom=False)
    else:
        groups = groups if groups else list(range(len(bars[0])))
        print(groups)
        print(labels)
        plt.xticks(
            [x_position + BAR_WIDTH for x_position in range(len(bars[0]))], groups
        )

    plt.legend()
    if save:
        plt.tight_layout()
        plt.savefig((path + ".jpg"), dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()


def plot_table(
    data, columns=None, rows=None, save=True, path="table", color="white", title="Table"
):
    fig, ax = plt.subplots(1, 1)
    ax.axis("tight")
    ax.axis("off")

    ax.table(
        cellText=data,
        colLabels=get_bold_text(columns) if columns else None,
        rowLabels=get_bold_text(rows) if rows else None,
        rowColours=[color] * len(data) if rows else None,
        colColours=[color] * len(data[0]) if columns else None,
        loc="center",
    )
    plt.title((title))
    if save:
        plt.tight_layout()
        plt.savefig((path + ".jpg"), dpi=300, bbox_inches="tight")
    # plt.close()
    plt.show()


def get_bold_text(text_list):
    bold_text = []
    for text in text_list:
        bold_text.append(f"$\\bf{text}$")
    return bold_text


def plot_confusion_matrix(cm, save=True, path="confusion matrix", model_name=""):
    cm_df = pd.DataFrame(
        cm, index=["Positive", "Negative"], columns=["Positive", "Negative"]
    )
    s = sns.heatmap(cm_df, annot=True, cmap="Blues", vmin=0, vmax=1)
    s = s.set(xlabel="True", ylabel="Predicted")
    plt.title(("Confusion Matrix - " + model_name))

    if save:
        plt.tight_layout()
        plt.savefig((path + ".jpg"), dpi=300, bbox_inches="tight")
    plt.close()
    plt.legend()
    plt.show()
