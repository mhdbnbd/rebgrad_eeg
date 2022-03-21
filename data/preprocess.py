from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.preprocessing import create_windows_from_events


# load data
def load_preprocess(subject_id, low_cut_hz, cross_session=True):
    """
      loads and preprocesses dataset BCI competition IV 2a
      Parameters:
      -----------------
      subject_id: subject to load (1..9)
      low_cut_hz: low cut frequency for filtering
      cross_session: if True data splits are from different sessions
      Returns:
      -----------------
      training set and validation set
    """
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_id)
    # preprocessing
    high_cut_hz = 38.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                     factor_new=factor_new, init_block_size=init_block_size)
    ]
    # Transform the data
    preprocess(dataset, preprocessors)

    # Cut into windows
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    # Calculate the trial start offset in samples.
    trial_start_offset_seconds = -0.5
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True
    )

    # split dataset
    if cross_session:
        splitted = windows_dataset.split('session')
        train_set = splitted['session_T']
        valid_set = splitted['session_E']
    else:
        splitted = windows_dataset.split(dict(train=[0, 1, 2, 3], valid=[4, 5]))
        train_set = splitted['train']
        valid_set = splitted['valid']

    # recreate the dataset in the shape data, labels
    # source/train
    source_x = []
    source_y = []

    for data in train_set:
        x = data[0]
        y = data[1]
        source_x.append(x)
        source_y.append(y)

    source_data = []
    for i in range(len(source_x)):
        source_data.append([source_x[i], source_y[i]])

    # target/valid
    target_x = []
    target_y = []

    for data in valid_set:
        x = data[0]
        y = data[1]
        target_x.append(x)
        target_y.append(y)

    target_data = []
    for i in range(len(source_x)):
        target_data.append([source_x[i], source_y[i]])

    return source_data, target_data
