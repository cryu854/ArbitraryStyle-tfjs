""" USAGE
python main.py train --content ./path/to/MSC0C0_dataset   \
                     --style ./path/to/WikiArt_dataset \
                     --batch 8 \
                     --debug True \
                     --validate_content ./path/to/validate/content.jpg \
                     --validate_style ./path/to/validate/style.jpg

python main.py inference --content ./path/to/content.jpg   \
                         --style ./path/to/style.jpg \
                         --alpha 1.0 \
                         --checkpoint ./path/to/pretrainind_model

"""

""" Convert pre-trained model to tensorflow-js model
tensorflowjs_converter --input_format=tf_saved_model --saved_model_tags=serve  checkpoint_dir/model checkpoint_dir/web_model
"""


import os
import argparse
import tensorflow as tf
from train import Trainer
from inference import Inferencer


CONTENT_WEIGHT = 1
STYLE_WEIGHT = 10
EXTRACT_LAYERS = ['block1_conv1',
                  'block2_conv1',
                  'block3_conv1',
                  'block4_conv1']
REFLECT_PADDING = False  # If True, using reflect padding for decoder and encoder, but the model cannot be able to be converted to tensorflow-js model.

LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 5e-5
NUM_EPOCHS = 6
BATCH_SIZE = 8

VALIDATE_CONTENT = './images/content/avril.jpg'
VALIDATE_STYLE = './images/style/udnie.jpg'
CONTENT_DATASET = './../../datasets/train2014/train2014'
STYLE_DATASET = './../../datasets/WikiArt'
CHECKPOINT_DIR = './checkpoint_dir'
RESULT_DIR = './results'
IMAGE_TYPE = ('jpg', 'jpeg', 'png', 'bmp')



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Arbitrary Style Transfer')
    parser.add_argument('command',
                         help="'train' or 'inference'")
    parser.add_argument('--content', required=False,
                         help='train: Content dataset to train; inference: Content image to stylize',
                         default=CONTENT_DATASET)
    parser.add_argument('--style', required=False,
                         help='train: Style dataset to train; inference: Specific style target',
                         default=STYLE_DATASET) 
    parser.add_argument('--alpha', required=False,type=float,
                         help='Image stylization intensity, value between 0 and 1',
                         default=1.0) 
    parser.add_argument('--checkpoint', required=False,
                         help='Checkpoints/model directory',
                         default=CHECKPOINT_DIR)
    parser.add_argument('--result', required=False,
                         help='Path to the stylized results',
                         default=RESULT_DIR)
    parser.add_argument('--batch', required=False, type=int,
                         help='Training batch size',
                         default=BATCH_SIZE)
    parser.add_argument('--debug', required=False, type=bool,
                         help='Whether to write the summary and stylize validate image during training',
                         default=True)
    parser.add_argument('--validate_content', required=False,
                         help='Training batch size',
                         default=VALIDATE_CONTENT)
    parser.add_argument('--validate_style', required=False,
                         help='Training batch size',
                         default=VALIDATE_STYLE)

    args = parser.parse_args()



    if args.command == "train":
        assert os.path.exists(args.content), 'Content dataset directory not found !'
        assert os.path.exists(args.style), 'Style dataset directory not found !'
        assert args.batch > 0
        assert NUM_EPOCHS > 0
        assert CONTENT_WEIGHT >= 0
        assert STYLE_WEIGHT >= 0
        assert LEARNING_RATE >= 0
        if args.debug:
            assert args.validate_content[-3:] in IMAGE_TYPE, 'Invalid validation content image'
            assert args.validate_style[-3:] in IMAGE_TYPE, 'Invalid validation style image'

        parameters = {
                'content_dir' : args.content,
                'style_dir' : args.style,
                'batch_size' : args.batch,
                'checkpoint_dir' : args.checkpoint,
                'debug' : args.debug,      
                'validate_content' : args.validate_content,
                'validate_style' : args.validate_style,      
                'style_weight': STYLE_WEIGHT,
                'content_weight' : CONTENT_WEIGHT,
                'extract_layers' : EXTRACT_LAYERS,
                'reflect_padding' : REFLECT_PADDING,
                'num_epochs' : NUM_EPOCHS,
                'learning_rate' : LEARNING_RATE,
                'lr_decay' : LEARNING_RATE_DECAY
        }

        trainer = Trainer(**parameters)
        trainer.train()


    elif args.command == "inference":
        assert args.content[-3:] in IMAGE_TYPE, 'Referenced content image not found !'
        assert args.style[-3:] in IMAGE_TYPE, 'Referenced style image not found !'  
        assert 0 <= args.alpha <= 1, 'The stylization intensity alpha must be between 0 and 1 '
        assert os.path.exists(args.checkpoint), 'Pretrained checkpoints/model path not found !'

        parameters = {
                'model_dir' : args.checkpoint,
                'result_dir' : args.result,
        }

        model = Inferencer(**parameters)
        content_image = model.preprocess_file(args.content)
        style_image = model.preprocess_file(args.style)
        stylized = model(content_image, style_image, args.alpha)
        model.save(stylized)


    else:
        print('Example usage : python main.py inference --content ./image.jpg --style ./style.jpg --checkpoint ./checkpoint/model')
        
        

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()