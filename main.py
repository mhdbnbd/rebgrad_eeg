from train import train

# Typical model training accross all subjects, 0-38Hz
for sub in [n+1 for n in range(9)]:
    train(batch_size=64, epochs=10, subject_id=sub, low_cut_hz=0)
    print("subject n ", sub)
