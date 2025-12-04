import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Set PyTorch memory allocation to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

modelName = 'speechBaseline4'

args = {}
args['outputDir'] = '/home/iseanbhanot/output_dir' + modelName
args['datasetPath'] = '/home/iseanbhanot/dataset/ptDecoder_ctc.pkl'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 32  # Reduced from 64 to fit in GPU memory
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)