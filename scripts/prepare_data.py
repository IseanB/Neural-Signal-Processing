"""
Prepare competition data by converting .mat files to pickle format.
Based on notebooks/formatCompetitionData.ipynb
"""
import os
import re
import pickle
import numpy as np
import scipy.io

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
    dat = scipy.io.loadmat(sessionPath)

    input_features = []
    transcriptions = []
    frame_lens = []
    n_trials = dat['sentenceText'].shape[0]

    # Collect area 6v tx1 and spikePow features
    for i in range(n_trials):
        # Get time series of TX and spike power for this trial
        # First 128 columns = area 6v only
        features = np.concatenate([dat['tx1'][0, i][:, 0:128], dat['spikePow'][0, i][:, 0:128]], axis=1)

        sentence_len = features.shape[0]
        sentence = dat['sentenceText'][i].strip()

        input_features.append(features)
        transcriptions.append(sentence)
        frame_lens.append(sentence_len)

    # Block-wise feature normalization
    blockNums = np.squeeze(dat['blockIdx'])
    blockList = np.unique(blockNums)
    blocks = []
    for b in range(len(blockList)):
        sentIdx = np.argwhere(blockNums == blockList[b])
        sentIdx = sentIdx[:, 0].astype(np.int32)
        blocks.append(sentIdx)

    for b in range(len(blocks)):
        feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1] + 1)], axis=0)
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        for i in blocks[b]:
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

    # Convert to session data format
    session_data = {
        'inputFeatures': input_features,
        'transcriptions': transcriptions,
        'frameLens': frame_lens
    }

    return session_data


def getDataset(fileName):
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
            p = re.sub(r'[0-9]', '', p)  # Remove stress
            if re.match(r'[A-Z]+', p):  # Only keep phonemes
                phonemes.append(p)

        # Add one SIL symbol at the end
        if addInterWordSymbol:
            phonemes.append('SIL')

        seqLen = len(phonemes)
        maxSeqLen = 500
        seqClassIDs = np.zeros([maxSeqLen]).astype(np.int32)
        seqClassIDs[0:seqLen] = [phoneToId(p) + 1 for p in phonemes]
        seqElements.append(seqClassIDs)

    newDataset = {}
    newDataset['sentenceDat'] = allDat
    newDataset['transcriptions'] = trueSentences
    newDataset['phonemes'] = seqElements

    timeSeriesLens = []
    phoneLens = []
    for x in range(len(newDataset['sentenceDat'])):
        timeSeriesLens.append(newDataset['sentenceDat'][x].shape[0])

        zeroIdx = np.argwhere(newDataset['phonemes'][x] == 0)
        phoneLens.append(zeroIdx[0, 0])

    newDataset['timeSeriesLens'] = np.array(timeSeriesLens)
    newDataset['phoneLens'] = np.array(phoneLens)
    newDataset['phonePerTime'] = newDataset['phoneLens'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32)
    return newDataset


def main():
    # Update these paths
    dataDir = '/home/iseanbhanot/dataset/competitionData'
    outputFile = '/home/iseanbhanot/dataset/ptDecoder_ctc.pkl'

    # Get session names from the train directory
    train_dir = os.path.join(dataDir, 'train')
    mat_files = [f.replace('.mat', '') for f in os.listdir(train_dir) if f.endswith('.mat')]
    sessionNames = sorted(mat_files)

    print(f"Found {len(sessionNames)} sessions")
    print(f"Processing in batches to conserve memory...")

    trainDatasets = []
    testDatasets = []
    competitionDatasets = []

    # Process in smaller batches to avoid memory issues
    batch_size = 4
    for batch_start in range(0, len(sessionNames), batch_size):
        batch_end = min(batch_start + batch_size, len(sessionNames))
        batch_sessions = sessionNames[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(sessionNames) + batch_size - 1)//batch_size}")

        for sessionName in batch_sessions:
            dayIdx = sessionNames.index(sessionName)
            print(f"  [{dayIdx + 1}/{len(sessionNames)}] {sessionName}")

            try:
                # Process train data
                train_path = os.path.join(dataDir, 'train', sessionName + '.mat')
                if os.path.exists(train_path):
                    trainDataset = getDataset(train_path)
                    trainDatasets.append(trainDataset)
                    del trainDataset  # Free memory immediately

                # Process test data
                test_path = os.path.join(dataDir, 'test', sessionName + '.mat')
                if os.path.exists(test_path):
                    testDataset = getDataset(test_path)
                    testDatasets.append(testDataset)
                    del testDataset  # Free memory immediately

                # Process competition hold-out data
                comp_path = os.path.join(dataDir, 'competitionHoldOut', sessionName + '.mat')
                if os.path.exists(comp_path):
                    dataset = getDataset(comp_path)
                    competitionDatasets.append(dataset)
                    del dataset  # Free memory immediately

            except MemoryError:
                print(f"  WARNING: Memory error processing {sessionName}, skipping...")
                continue
            except Exception as e:
                print(f"  ERROR processing {sessionName}: {e}")
                raise

        # Force garbage collection after each batch
        import gc
        gc.collect()
        print(f"  Batch complete. Memory freed.")

    # Save to pickle file
    allDatasets = {
        'train': trainDatasets,
        'test': testDatasets,
        'competition': competitionDatasets
    }

    print(f"\nSaving to {outputFile}")
    with open(outputFile, 'wb') as handle:
        pickle.dump(allDatasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done! Processed:")
    print(f"  - {len(trainDatasets)} train datasets")
    print(f"  - {len(testDatasets)} test datasets")
    print(f"  - {len(competitionDatasets)} competition datasets")


if __name__ == '__main__':
    main()
