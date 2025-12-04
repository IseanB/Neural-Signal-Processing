"""
Memory-efficient data preparation for low-memory systems.
Processes one session at a time and saves incrementally.
"""
import os
import re
import pickle
import numpy as np
import scipy.io
import gc

# Download required NLTK data for g2p_en
import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)

from g2p_en import G2p

# Phone definitions
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
PHONE_DEF_SIL = PHONE_DEF + ['SIL']

g2p = G2p()

def phoneToId(p):
    return PHONE_DEF_SIL.index(p)


def loadFeaturesAndNormalize(sessionPath):
    """Load and normalize features with minimal memory footprint."""
    dat = scipy.io.loadmat(sessionPath)

    input_features = []
    transcriptions = []
    n_trials = dat['sentenceText'].shape[0]

    # Collect features
    for i in range(n_trials):
        # Only keep area 6v (first 128 columns)
        features = np.concatenate([dat['tx1'][0, i][:, 0:128], dat['spikePow'][0, i][:, 0:128]], axis=1).astype(np.float32)
        sentence = dat['sentenceText'][i].strip()

        input_features.append(features)
        transcriptions.append(sentence)

    # Block-wise feature normalization
    blockNums = np.squeeze(dat['blockIdx'])
    blockList = np.unique(blockNums)
    blocks = []
    for b in range(len(blockList)):
        sentIdx = np.argwhere(blockNums == blockList[b])
        sentIdx = sentIdx[:, 0].astype(np.int32)
        blocks.append(sentIdx)

    for b in range(len(blocks)):
        feats = np.concatenate([input_features[i] for i in blocks[b]], axis=0)
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        del feats  # Free memory
        gc.collect()

        for i in blocks[b]:
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

    session_data = {
        'inputFeatures': input_features,
        'transcriptions': transcriptions,
    }

    # Clear the loaded mat file from memory
    del dat
    gc.collect()

    return session_data


def getDataset(fileName):
    """Process a single session file."""
    session_data = loadFeaturesAndNormalize(fileName)

    allDat = []
    trueSentences = []
    seqElements = []

    for x in range(len(session_data['inputFeatures'])):
        allDat.append(session_data['inputFeatures'][x])
        trueSentences.append(session_data['transcriptions'][x])

        thisTranscription = str(session_data['transcriptions'][x]).strip()
        thisTranscription = re.sub(r'[^a-zA-Z\- \']', '', thisTranscription)
        thisTranscription = thisTranscription.replace('--', '').lower()
        addInterWordSymbol = True

        phonemes = []
        for p in g2p(thisTranscription):
            if addInterWordSymbol and p == ' ':
                phonemes.append('SIL')
            p = re.sub(r'[0-9]', '', p)
            if re.match(r'[A-Z]+', p):
                phonemes.append(p)

        if addInterWordSymbol:
            phonemes.append('SIL')

        seqLen = len(phonemes)
        maxSeqLen = 500
        seqClassIDs = np.zeros([maxSeqLen], dtype=np.int32)
        seqClassIDs[0:seqLen] = [phoneToId(p) + 1 for p in phonemes]
        seqElements.append(seqClassIDs)

    newDataset = {
        'sentenceDat': allDat,
        'transcriptions': trueSentences,
        'phonemes': seqElements
    }

    timeSeriesLens = []
    phoneLens = []
    for x in range(len(newDataset['sentenceDat'])):
        timeSeriesLens.append(newDataset['sentenceDat'][x].shape[0])

        zeroIdx = np.argwhere(newDataset['phonemes'][x] == 0)
        phoneLens.append(zeroIdx[0, 0])

    newDataset['timeSeriesLens'] = np.array(timeSeriesLens, dtype=np.int32)
    newDataset['phoneLens'] = np.array(phoneLens, dtype=np.int32)
    newDataset['phonePerTime'] = newDataset['phoneLens'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32)

    # Clear session_data
    del session_data
    gc.collect()

    return newDataset


def main():
    dataDir = '/home/iseanbhanot/dataset/competitionData'
    outputFile = '/home/iseanbhanot/dataset/ptDecoder_ctc.pkl'

    # Get session names
    train_dir = os.path.join(dataDir, 'train')
    mat_files = [f.replace('.mat', '') for f in os.listdir(train_dir) if f.endswith('.mat')]
    sessionNames = sorted(mat_files)

    print(f"Found {len(sessionNames)} sessions")
    print(f"Processing with aggressive memory management...")
    print(f"Available memory: {os.popen('free -h | grep Mem').read().strip()}\n")

    trainDatasets = []
    testDatasets = []
    competitionDatasets = []

    for idx, sessionName in enumerate(sessionNames):
        print(f"[{idx + 1}/{len(sessionNames)}] {sessionName}", end='', flush=True)

        try:
            # Process train data
            train_path = os.path.join(dataDir, 'train', sessionName + '.mat')
            if os.path.exists(train_path):
                print(" (train)", end='', flush=True)
                trainDataset = getDataset(train_path)
                trainDatasets.append(trainDataset)
                del trainDataset
                gc.collect()

            # Process test data
            test_path = os.path.join(dataDir, 'test', sessionName + '.mat')
            if os.path.exists(test_path):
                print(" (test)", end='', flush=True)
                testDataset = getDataset(test_path)
                testDatasets.append(testDataset)
                del testDataset
                gc.collect()

            # Process competition hold-out data
            comp_path = os.path.join(dataDir, 'competitionHoldOut', sessionName + '.mat')
            if os.path.exists(comp_path):
                print(" (comp)", end='', flush=True)
                dataset = getDataset(comp_path)
                competitionDatasets.append(dataset)
                del dataset
                gc.collect()

            print(" ✓")

        except MemoryError as e:
            print(f" ✗ MEMORY ERROR - Instance may crash!")
            print(f"Try reducing batch_size or upgrading instance memory")
            raise
        except Exception as e:
            print(f" ✗ ERROR: {e}")
            raise

    # Save to pickle file
    print(f"\nSaving to {outputFile}...")
    allDatasets = {
        'train': trainDatasets,
        'test': testDatasets,
        'competition': competitionDatasets
    }

    with open(outputFile, 'wb') as handle:
        pickle.dump(allDatasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Clear memory
    del allDatasets, trainDatasets, testDatasets, competitionDatasets
    gc.collect()

    print(f"Done!")


if __name__ == '__main__':
    main()
