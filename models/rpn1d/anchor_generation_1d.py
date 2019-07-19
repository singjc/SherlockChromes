import numpy as np

def generate_anchors_1d(
    sequence_length=2372,
    base_half_width=7,
    width_delta=0,
    num_deltas=0,
    stride=1):
    anchors = []

    for i in range(0, sequence_length, stride):
        for j in range(num_deltas + 1):
            half_width = (base_half_width + (j * width_delta))
            anchors.append([float((i - half_width)), float((i + half_width))])

    return np.array(anchors), num_deltas + 1

if __name__ == "__main__":
    anchors, num_anchors = generate_anchors_1d()
    print(anchors.shape)
