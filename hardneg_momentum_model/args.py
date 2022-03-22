import argparse

parser = argparse.ArgumentParser(description='Parameters for Training/Testing/Evaluating with the Generalizable Coherence Model')
parser.add_argument('--train_file', help='Provide path to pickle file with training data. Refer to README for format.')
parser.add_argument('--dev_file', help='Provide path to pickle file with development data. Refer to README for format.')
parser.add_argument('--test_file', help='Provide path to pickle file with test data. Refer to README for format.')
parser.add_argument('--eval_file', help='Provide path to pickle file with documents you want to evaluate. Refer to README for format.')
parser.add_argument('--model_size', default='base', help='Specify XLNet model size. Note that the model has only been tested with XLNet-base.')
parser.add_argument('--lr_start', type=float, default=5e-6, help='Starting learning rate')
parser.add_argument('--lr_end', type=float, default=1e-6, help='Final learning rate to anneal to')
parser.add_argument('--lr_anneal_epochs', type=int, default=50, help='Number of epochs/times over which to anneal the learning rate')
parser.add_argument('--eval_interval', type=int, default=1000, help='Frequency of evaluation on the dev data and the LR scheduler anneal call, in number of training steps. Note that you may need to account for batch size if counting by training samples.')
parser.add_argument('--seed', type=int, default=100, help='Set the random seed')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--margin', type=float, default=0.1, help='Margin for pairwise/max-margin or contrastive loss.')
parser.add_argument('--model_description', default='model_checkpoint_', help='Model description to be added to saved checkpoint filename.')
parser.add_argument('--data_type', default='multiple', help='Whether the input has a single negative or multiple negatives. Refer to README for format.')
parser.add_argument('--num_negs', type=int, default=5, help='Specify the number of negative samples per positive sample for training. This should be 2+ for the contrastive models. Note that the model was tested with 5 negatives.')
parser.add_argument('--num_rank_negs', type=int, default=50, help='Total number of negative samples pe positive sample that need to be ranked. The extra negatives will be used for hard negative mining and the top N ranked negative samples (number of negatives specified by --num_negs) will be used for training.')
parser.add_argument('--train_steps', type=int, default=200, help='Number of steps to train for before mining hard negatives for the next training steps.')
parser.add_argument('--momentum_coefficient', type=float, default=0.9999999, help='Momentum coefficient for updating the auxiliary encoder.')
parser.add_argument('--queue_size', type=int, default=1000, help='Size of the global negative sample queue.')
parser.add_argument('--contrastive_loss_weight', type=float, default=0.85, help='Weight of the original contrastive loss to the momentum loss. Note that the momentum loss weight will be 1 minus the value specified here.')
parser.add_argument('--max_len', type=int, default=600, help='Maximum number of tokens (based on XLNet tokenizer) in the document. Tokens will be padded or excess will be truncated. Ensure the data is preprocessed accordingly so that truncation does not affect the task.')
parser.add_argument('--pretrained_model', default='', help='Path to pre-trained model. Load a pre-trained model for testing or fine-tuning.')




