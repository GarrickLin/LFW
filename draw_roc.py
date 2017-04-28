import matplotlib.pyplot as plt
plt.style.use('bmh')
import os


def plot_roc(filename):
    tpr = []
    fpr = []
    with open(filename) as f:
        next(f)
        for line in f:
            raw = line.strip().split(',')
            fpr.append(float(raw[-1]))
            tpr.append(float(raw[-2]))
            
    fig, ax = plt.subplots(1, 1)
    plt.plot([0, 1], color='k', linestyle=':')

    plt.plot(fpr, tpr)
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.ylim(ymin=0,ymax=1)
    plt.xlim(xmin=0, xmax=1)

    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
    plt.minorticks_on()
    fig.savefig(os.path.join("data/", "roc.png"))    
    

if __name__ == "__main__":
    plot_roc("data/roc.txt")