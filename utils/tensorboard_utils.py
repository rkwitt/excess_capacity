import json
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData

def tb_to_json(path: str):
    """Converts the tensorboard event files from one run to a single json file. The output is a dictionary with keys  'hparam', 'final', 'scalar', 'hist', 'norms'.

    'hparam' --> dict of hyperparameters used for training
    'final' --> dict of loss/acc after training
    'scalar' --> dict of quantities (loss/acc) tracked during training (epochs and values)
    'hist' -->  histograms (margins, normalized margins) after training
    'norms' --> dict of quantities (lip/norm) tracked during training for each layer (SGD steps and values)
    """
    summary = {}

    # identify logfile with hyper parameters and final accuracy etc
    for file in os.listdir(path):
        if file.replace('.','').isdigit():
            hparams_dir = file
            break
    hparams_log = os.path.join(path,hparams_dir)
    hparams_log = os.path.join(hparams_log, os.listdir(hparams_log)[0])

    #load logfile    
    event_acc = EventAccumulator(hparams_log)
    event_acc.Reload()

    # get hyper paramters
    s = event_acc._plugin_to_tag_to_content['hparams']["_hparams_/session_start_info"]
    hdata = HParamsPluginData.FromString(s).session_start_info.hparams
    hparam_dict = {key: hdata[key].ListFields()[0][1] for key in hdata.keys()}
    summary['hparam'] = hparam_dict

    #get final loss, ecc, ...
    final = {}
    for scalar in event_acc.Tags()['scalars']:
        values = event_acc.Scalars(scalar)
        final[scalar] = values[0].value
        summary['final'] = final

    #identify logfile with scalars:
    for file in os.listdir(path):
        if 'events.out.tfevents.' in file:
            logfile = os.path.join(path, file)
            break

    #load logfile    
    event_acc = EventAccumulator(logfile)
    event_acc.Reload()

    # get scalars (loss curve, acc curve, etc)
    scalars = {}
    for scalar in event_acc.Tags()['scalars']:
        item = event_acc.Scalars(scalar)
        steps = [v.step for v in item]
        values = [v.value for v in item]
        scalars[scalar] = {'steps': steps, 'values': values}
        summary['scalar'] = scalars

    # get margin distribution
    hists = {}
    for hist in event_acc.Tags()['histograms']:
        item = event_acc.Histograms(hist)
        #only get last histogram --> [-1]
        values = item[-1].histogram_value
        hists[hist] = {'buckets': values.bucket_limit, 'values': values.bucket}
    summary['hist'] = hists

    # get lipschitz consts and dists
    conds = os.listdir(path)
    norms = {}
    for cond in conds:
        if 'Condition' in cond:
            quant = cond.split('_')[1]
            layer = cond.split('_')[2]
            if not quant in norms.keys():
                norms[quant] = {}
            if not layer in norms[quant].keys():
                norms[quant][layer] = {}

            
            event_file = os.path.join(path, cond)
            event_file = os.path.join(event_file, os.listdir(event_file)[0])
            
            event_acc = EventAccumulator(event_file)
            event_acc.Reload()
            
            for scalar in event_acc.Tags()['scalars']:
                item = event_acc.Scalars(scalar)
                steps = [v.step for v in item]
                values = [v.value for v in item]
            norms[quant][layer] = {'steps': steps, 'values': values}
    summary['norms'] =norms

    with open(path + '/log.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)