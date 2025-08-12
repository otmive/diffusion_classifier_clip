import pandas as pd
import argparse
import os
import numpy as np 

train_classes = [
    "red sphere",
    "green sphere",
    "purple sphere",
    "cyan sphere",
    "gray sphere",
    "blue sphere",
    "yellow sphere",
    "gray cube",
    "blue cube",
    "brown cube",
    "red cylinder",
    "green cylinder",
    "purple cylinder",
    "cyan cylinder",
    "purple cone",
    "brown cone",
    "yellow cone"
]

def get_rel_prompt_folder(data_type):
    type_dict = {'idval':'train_rel_prompts',
                 'idtest':'train_rel_prompts',
                 'val':'beth_rel_prompts_val',
                 'test':'test_rel_prompts',
                 'idval_gen':'train_rel_prompts_gen',
                 'idtest_gen': 'train_rel_prompts_gen',
                 'val_gen': 'beth_rel_prompts_val_gen',
                 'test_gen': 'test_rel_prompts_gen'}
    
    return type_dict[data_type]

def get_prompt_folder(data_type):
    type_dict = {'idval':'train_two_obj_prompts',
                 'idtest':'train_two_obj_prompts',
                 'val':'beth_two_obj_prompts',
                 'test':'test_two_obj_prompts',
                 'idval_gen':'train_two_obj_prompts_gen',
                 'idtest_gen': 'train_two_obj_prompts_gen',
                 'val_gen': 'gen_two_obj_prompts',
                 'test_gen': 'test_two_obj_prompts_gen'}
    
    return type_dict[data_type]

def get_hard_neg_colour(pred,true,file):
    file = file.split("/")[3]
    first_colour = file.split("_")[0]
    first_shape = file.split("_")[1]
    second_colour = file.split("_")[2] 
    second_shape = file.split("_")[3]
    if pred.split()[0] == second_colour and pred.split()[1] == first_shape:
        return True
    else:
        return False
    
def get_hard_neg_shape(pred,true,file):
    file = file.split("/")[3]
    first_colour = file.split("_")[0]
    first_shape = file.split("_")[1]
    second_colour = file.split("_")[2] 
    second_shape = file.split("_")[3].split(".")[0]
    if pred.split()[1] == second_shape and pred.split()[0] == first_colour:
        return True
    else:
        return False
    
def get_hard_neg_leftright(pred, true):
    if 'right' in true:
        hard_neg = true.split()[0] + " left " + true.split()[2]
    elif 'left' in true:
        hard_neg = true.split()[0] + " right " + true.split()[2]
    if pred == hard_neg:
        return True
    else:
        return False

def get_hard_neg_shapeswap(pred, true):

    hard_neg = true.split()[2] + " " + true.split()[1] + " " + true.split()[0]

    if pred == hard_neg:
        return True
    else:
        return False

def same_colour(pred, true):
    if pred.split()[0] == true.split()[0]:
        return True
    else:
        return False

def same_shape(pred, true):
    if pred.split()[1] == true.split()[1]:
        return True
    else:
        return False
    
def get_text_val(val):
    prompts = pd.read_csv('prompts/clevr_prompts_cone.csv')
    text_prompt = prompts['class_name'].tolist()[val]
    return text_prompt 

def print_acc(output_file, errors):
    print(errors)
    out = pd.read_csv(output_file)
    count = 0
    colour_count=0
    shape_count=0
    for i in range(len(out)):
        if out['pred'][i] == out['true'][i]:
            count+=1
        else:
            if type(out['pred'][i]) ==np.int64:
                pred = get_text_val(out['pred'][i])
                true = get_text_val(out['true'][i])
            else:
                true = out['true'][i]
                pred = out['pred'][i]
            if same_colour(true, pred):
                colour_count += 1
            if same_shape(true, pred):
                shape_count += 1        

        
    acc = round(count*100/len(out), 4)
    print(acc)

    if errors == True:
        colour = round(colour_count/(len(out)),4)
        print("same colour")
        print(colour*100)
        shape = round(shape_count/(len(out)),4)
        print("same shape")
        print(shape*100)
        both = round((len(out)-count-shape_count-colour_count)/(len(out)), 4)
        print("both shape colour wrong")
        print(both*100)


def print_acc_two(output_folder, errors, data_type):
    count = 0
    leng = 0
    hn_colour = 0
    hn_shape = 0
    seen = 0
    unseen = 0
    prompt_folder = get_prompt_folder(data_type)
    for file in os.listdir(output_folder):
        out = pd.read_csv(output_folder+"/"+file)
        for i in range(len(out)):
            if out['pred'][i] == out['true'][i]:
                count+=1
            if errors == True:
                prompt_file = 'prompts/two_object_prompts/' + prompt_folder + "/" + file
                prompts = pd.read_csv(prompt_file)['class_name'].tolist()
                if type(out['pred'][i]) ==np.int64:
                    pred = prompts[out['pred'][i]]
                    true = prompts[out['true'][i]]
                else:
                    true = out['true'][i]
                    pred = out['pred'][i]
                if get_hard_neg_colour(pred,true,prompt_file) == True:
                    hn_colour += 1
                if get_hard_neg_shape(pred,true,prompt_file) == True:
                    hn_shape += 1

                if pred in train_classes:
                    seen+=1
                else:
                    unseen+=1
                
        
        leng += len(out)
    print(round(count*100/leng,4))

    if errors == True:
        hard_neg_colour = hn_colour/(leng)
        hard_neg_shape = hn_shape/(leng)
        seen_perc = seen/(leng)
        unseen_perc = unseen/(leng)
        other = (leng-hn_colour-hn_shape-count)/(leng)
        print("colour of second shape:")
        print(hard_neg_colour*100)
        print("shape of second shape:")
        print(hard_neg_shape*100)
        print("other prompt")
        print(other*100)
        # print("seen:")
        # print(seen_perc)
        # print("unseen:")
        # print(unseen_perc)

def print_acc_rel(output_folder, errors, data_type):
    count = 0
    leng = 0
    hn_leftright = 0
    hn_shapeswap = 0
    seen = 0
    unseen = 0
    prompt_folder = get_rel_prompt_folder(data_type)
    for file in os.listdir(output_folder):
        out = pd.read_csv(output_folder+"/"+file)
        for i in range(len(out)):
            if out['pred'][i] == out['true'][i]:
                count+=1
            if errors == True:
                if 'output' in file:
                    file = file.replace(" ", "_")
                    file = file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2] + ".csv"
                prompt_file = 'prompts/rel_prompts/' + prompt_folder + "/" + file
                prompts = pd.read_csv(prompt_file)['class_name'].tolist()
                if type(out['pred'][i]) ==np.int64:
                    pred = prompts[out['pred'][i]]
                    true = prompts[out['true'][i]]
                else:
                    true = out['true'][i]
                    pred = out['pred'][i]
                if get_hard_neg_leftright(pred,true) == True:
                    hn_leftright += 1
                if get_hard_neg_shapeswap(pred,true) == True:
                    hn_shapeswap += 1

                if pred in train_classes:
                    seen+=1
                else:
                    unseen+=1
                
        
        leng += len(out)
    print(round(count*100/leng,4))

    if errors == True:
        hard_neg_leftright = hn_leftright/(leng)
        hard_neg_shapeswap = hn_shapeswap/(leng)
        seen_perc = seen/(leng)
        unseen_perc = unseen/(leng)
        other = (leng-count-hn_leftright-hn_shapeswap)/(leng)
        print("left right wrong way:")
        print(hard_neg_leftright*100)
        print("shapes wrong way:")
        print(hard_neg_shapeswap*100)
        print("other prompt")
        print(other*100)
        # print("seen:")
        # print(seen_perc)
        # print("unseen:")
        # print(unseen_perc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', required=True, help='Path to output file of CLIP results')
    parser.add_argument('--dataset', default='single' )
    parser.add_argument('--errors', default=False, type=bool)
    parser.add_argument('--datatype', default=None)
    args = parser.parse_args()
    if args.dataset == 'single':
        print_acc(args.output_file, args.errors)
    elif args.dataset == 'two':
        print_acc_two(args.output_file, args.errors, args.datatype)
    elif args.dataset == 'rel':
        print_acc_rel(args.output_file, args.errors, args.datatype)