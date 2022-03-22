import argparse

parser = argparse.ArgumentParser(description='Parameters for Training/Testing/Evaluating with the Generalizable Coherence Model')
parser.add_argument('--train_file', help='Provide path to pickle file with training data. Refer to README for format.')
parser.add_argument('--dev_file', help='Provide path to pickle file with development data. Refer to README for format.')
parser.add_argument('--test_file', help='Provide path to pickle file with test data. Refer to README for format.')
parser.add_argument('--eval_file', help='Provide path to pickle file with documents you want to evaluate. Refer to README for format.')
parser.add_argument('--model_size', default='base', help='Specify XLNet model size. Note that the model has only been tested with XLNet-base.')
parser.add_argument('--lr_start', type=float, default=5e-6, help='Starting learning rate')
parser.add_argument('--lr_end', type=float, default=1e-6, help='Final learning rate to anneal to')
parser.add_argument('--lr_anneal_epochs', type=int, default=10, help='Number of epochs/times over which to anneal the learning rate')
parser.add_argument('--eval_interval', type=int, default=1000, help='Frequency of evaluation on the dev data and the LR scheduler anneal call, in number of training steps. Note that you may need to account for batch size.')
parser.add_argument('--seed', type=int, default=100, help='Set the random seed')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--margin', type=float, default=0.1, help='Margin for pairwise/max-margin or contrastive loss.')
parser.add_argument('--model_description', default='model_checkpoint_', help='Model description to be added to saved checkpoint filename.')
parser.add_argument('--data_type', default='multiple', help='Whether the input has a single negative or multiple negatives. Refer to README for format.')
parser.add_argument('--num_negs', type=int, default=1, help='Specify the number of negative samples for each positive sample. This should be 1 for pairwise, 2+ for contrastive, and in the order of 25+ for the momentum model to enable hard negative mining. Use in combination with ')
parser.add_argument('--max_len', type=int, default=600, help='Maximum number of tokens (based on XLNet tokenizer) in the document. Tokens will be padded or excess will be truncated. Ensure the data is preprocessed accordingly so that truncation does not affect the task.')
parser.add_argument('--pretrained_model', default='', help='Path to pre-trained model. Load a pre-trained model for testing or fine-tuning.')




