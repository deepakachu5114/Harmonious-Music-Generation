# Bayesic: This program takes in a number of parent and descendant nodes
# from the command line and, according to their assigned probabilities,
# samples the Bayesian network a fixed number of times, appending a note
# for each node that takes on the value 1 in the sample.

from music21 import *
import random
from tkinter import *
from tkinter.ttk import *

# each node will be on or off with prob p
import random
import networkx as nx
import matplotlib.pyplot as plt
# Initialize a directed graph to represent the Bayesian network
def bay():
    G = nx.DiGraph()

    def bernoulli(p):
        event = random.random() < p
        if event:
            return 1
        else:
            return 0

    # begin interacting with user

    parents = []
    descendants = {}
    tiers = {}
    probs = {}
    freqs = {}

    inp = "Beginning..."
    print(inp)
    emo = input("Enter emotion(h for happy, s for sad, a for angry): ")

    # get all parents

    """happy = [528, 261, 391, 138, 783, 146, 659, 277, 164, 293, 554, 46, 48, 54, 440, 184, 61, 65, 195, 69, 73, 329, 587, 82, 466, 219, 92, 97, 739, 233, 109, 493, 880, 369, 116, 246, 123]
    sad = [432, 97, 195, 164, 261, 41, 146, 246, 219, 130, 293, 329, 123, 109]
    angry = [261, 391, 523, 659, 32, 164, 41, 554, 43, 46, 48, 440, 184, 698, 65, 329, 587, 82, 466, 92, 97, 739, 493, 622, 369]"""

    happy = [261.93, 329.63, 392, 493.88, 587.33]
    sad = [220, 261.63, 329.63, 392, 493.88]
    angry = [311.13, 349.23, 466.16, 493.88, 523.23, 739.99]

    number_of_parents = 5
    prob = (random.uniform(20, 40)) / 100

    while True:
        number_of_parents -= 1
        parent = f"{emo}-{number_of_parents +1}"
        probs[parent] = prob
        parents.append(parent)
        if number_of_parents == 0:
            break

    tiers[0] = parents

    # get all descendants

    cont = "yes"
    tierno = 1

    for desc, plist in descendants.items():
        if len(plist) == 1:
            prob0_label = desc + "_" + plist[0] + "0"
            prob1_label = desc + "_" + plist[0] + "1"

            prob0 = input("Enter the probability of " + desc + "|" + plist[0] + "= 0.")
            prob1 = input("Enter the probability of " + desc + "|" + plist[0] + "= 1.")

            probs[prob0_label] = float(prob0)
            probs[prob1_label] = float(prob1)

        else:
            prob00_label = desc + "_" + plist[0] + "0" + plist[1] + "0"
            prob01_label = desc + "_" + plist[0] + "0" + plist[1] + "1"
            prob10_label = desc + "_" + plist[0] + "1" + plist[1] + "0"
            prob11_label = desc + "_" + plist[0] + "1" + plist[1] + "1"

            prob00 = input(
                "Enter the probability of "
                + desc
                + "|"
                + plist[0]
                + "= 0,"
                + plist[1]
                + "= 0."
            )
            prob01 = input(
                "Enter the probability of "
                + desc
                + "|"
                + plist[0]
                + "= 0,"
                + plist[1]
                + "= 1."
            )
            prob10 = input(
                "Enter the probability of "
                + desc
                + "|"
                + plist[0]
                + "= 1,"
                + plist[1]
                + "= 0."
            )
            prob11 = input(
                "Enter the probability of "
                + desc
                + "|"
                + plist[0]
                + "= 1,"
                + plist[1]
                + "= 1."
            )

            probs[prob00_label] = float(prob00)
            probs[prob01_label] = float(prob01)
            probs[prob10_label] = float(prob10)
            probs[prob11_label] = float(prob11)

    # assign frequencies to each parent

    for parent in parents:
        if emo == "h":
            val = random.choice(happy)
            freqs[parent] = val
            happy.remove(val)
        elif emo == "s":
            val = random.choice(sad)
            freqs[parent] = val
            sad.remove(val)
        elif emo == "a":
            val = random.choice(angry)
            freqs[parent] = val
            angry.remove(val)

    ngens = 0
    if emo == "h":
        ngens = 500
    elif emo == "a":
        ngens = 800
    else:
        ngens = 200

    # start making music!
    for parent in parents:
        G.add_node(parent)
    for desc, plist in descendants.items():
        for parent in plist:
            G.add_edge(parent, desc)

    # Draw the Bayesian network
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1000,
        node_color="lightblue",
        font_size=10,
        font_color="black",
        font_weight="bold",
    )
    plt.title("Bayesian Network")
    plt.show()

    waterfalls = stream.Stream()
    amp = 90
    amp_mod = 0.9  # 90% of volume for each tier
    w0 = 0.3  # if parent state 0, this much of it goes into average
    w1 = 1
    nodes = parents + list(descendants.keys())
    nodestates = {}

    for i in range(int(ngens)):
        if i > 0:
            r = note.Rest()  # a rest between each generation
            if emo == "a":
                r.duration.quarterLength = 0.1
            elif emo == "h":
                r.duration.quarterLength = 0.25
            else:
                r.duration.quarterLength = 0.4
            waterfalls.append(r)

        # sampling time!
        print("Sample ", i)

        for parent in parents:
            prob = probs[parent]
            state = bernoulli(prob)
            nodestates[parent] = state
            if state != 0:  # if parent on
                f = note.Note()
                f.pitch.frequency = float(freqs[parent])
                f.volume.velocity = amp
                if emo == "h":
                    f.duration.quarterLength = 0.25
                elif emo == "a":
                    f.duration.quarterLength = 0.2
                else:
                    f.duration.quarterLength = 1
                waterfalls.append(f)

        for tierno, tiernodes in tiers.items():
            if tierno == 0:
                continue

            for node in tiernodes:
                node_ps = descendants[node]  # all parents this node is conditioned on
                cond = ""
                nodef = 0

                for p in node_ps:
                    cond += p + str(nodestates[p])
                    if nodestates[p] == 0:
                        weight = w0
                    else:
                        weight = w1

                    nodef += float(freqs[p]) * weight

                nodef /= len(node_ps)  # weighted average
                label = node + "_" + cond

                prob_node = probs[label]
                freqs[node] = nodef
                state = bernoulli(prob_node)
                nodestates[node] = state

                if state != 0:
                    f = note.Note()
                    f.pitch.frequency = nodef
                    f.volume.velocity = amp * amp_mod**tierno
                    if emo == "h":
                        f.duration.quarterLength = 0.25
                    elif emo == "a":
                        f.duration.quarterLength = 0.2
                    else:
                        f.duration.quarterLength = 1
                    waterfalls.append(f)

        # nodestates is the whole sample now!
        print(nodestates)

    filename = (
        str(len(parents))
        + "_"
        + str(ngens)
        + "_"
        + str(len(list(descendants.keys())))
        + ".mid"
    )
    waterfalls.write(
        "midi",
        fp=f"/home/deepakachu/Desktop/DMS/project_final/generated_music/{filename}-{emo}.mid",
    )



def main():
    bay()
