import os
import tqdm
import tensorflow as tf

import CYCLEGAN.utils as utils
from CYCLEGAN.GANs.model import cycleGAN
from CYCLEGAN.GANs.net_utils import Checkpoint, tensorboard_output, data_loader, save_generations, setup_gpus

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = {
    'NAME': 'roadfighter-lvl2',
    'MODEL_PATH': 'cyclegan/models/outputs',
    'EPOCHS': 10,
    'DATA_LENGTH': 30000,
    'LEARNING_RATE': 0.0002,
    'START_DECAY': 3,
    'LS_GAN': True,
    'TYPE_LOSS': 'l1',
    'IMG_SIZE': 256,
    'LABEL_SMOOTHING': 0.9,
    'TYPE_NORM': 'instance',
    'BETA1': 0.5,
    'LAMBDA1': 10,
    'LAMBDA2': 0.5,
    'EVAL_PATH': '',
    'BATCH_SIZE': 1,
    'DATA_DIR': 'cyclegan/train_data/roadfighter-lvl2',
    'IMG_WIDTH': 84,
    'IMG_HEIGHT': 84,
    'IMG_DEPTH': 3,
    'DEBUG': True
}


def train(config):
    log_dir = '{}/{}/train/logs'.format(config['MODEL_PATH'], config['NAME'])
    utils.mdir(log_dir)
    utils.spit_json('{}/meta.json'.format(log_dir), config)

    setup_gpus()

    model = cycleGAN(config)

    epoch_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    checkpoint = Checkpoint(dict(G_A2B=model.G,
                                 G_B2A=model.F,
                                 D_A=model.Da,
                                 D_B=model.Db,
                                 G_optimizer=model.opt1,
                                 D_optimizer=model.opt2,
                                 ep_cnt=epoch_counter), os.path.join(log_dir, 'ckpts'), max_to_keep=3)
    try:
        checkpoint.restore().assert_existing_objects_matched()
    except Exception as e:
        print(e)

    if config['DEBUG']:
        debug_dir = os.path.join(log_dir, 'train_debug')
        utils.mdir(debug_dir)

    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'summaries'))

    # data loader
    domainA = data_loader(config, 'lvl2').build_dataset()
    domainB = data_loader(config, 'lvl1').build_dataset()
    zip_dataset = tf.data.Dataset.zip((domainA, domainB))

    step = 0

    for e in tqdm.trange(config['EPOCHS'], desc='Epoch iterator'):
        if e < epoch_counter:
            continue

        epoch_counter.assign_add(1)

        for a, b in tqdm.tqdm(zip_dataset, desc='Step iterator', total=config.get('DATA_LENGTH')):
            gen_summaries, dis_summaries = model.train_step(a, b)

            if step % 100 == 0 and config['DEBUG']:
                save_generations(model.supervise_generations(a, b), debug_dir, step)

            if step % 200 == 0:
                with summary_writer.as_default():
                    tensorboard_output(gen_summaries, step, name='generator_losses')
                    tensorboard_output(dis_summaries, step, name='discriminator_losses')
                    tensorboard_output({'learning_rate': model.schedule_gen.current_learning_rate},
                                        step, name='learning rate schedule')

            step += 1

            if step % config.get['DATA_LENGTH'] == 0:
                summary_writer.flush()

        # sanity check
        if step == config['DATA_LENGTH']*config['EPOCHS']:
            break

        checkpoint.save(e)
    print('Training Complete!')
    return None


def serve_checkpoint(config):
    out_dir = os.path.join(config['MODEL_PATH'], config['NAME'])
    train_log_dir = '{}/train/logs/'.format(out_dir)

    setup_gpus()

    model = cycleGAN(config)
    architecture = dict(G_A2B=model.G)
    ckpt = Checkpoint(architecture, os.path.join(train_log_dir, 'ckpts'), max_to_keep=3)

    try:
        ckpt.restore().assert_existing_objects_matched()
    except Exception as e:
        print(e)

    serve_call = model.G.serve.get_concrete_function(tf.TensorSpec([None, 84, 84, 3]), tf.float32)
    tf.saved_model.save(model.G, os.path.join(train_log_dir, 'frozen'), signatures=serve_call)
    return None


if __name__ == '__main__':
    train(config)
