import clip
import torch
from PIL import Image
import glob
import os
import pandas as pd
import argparse

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def predict(img_path, output_file, model_path):



    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)
    if model_path != None:
        model.load_state_dict(torch.load(model_path))

    class_names = []
    for folder in os.listdir(img_path):
        class_names.append(folder)
    print(class_names)

    classes = pd.read_csv("cobi2_datasets/single_object/single_prompts.csv")['class_name'].tolist()
    #candidate_names = ["blue cone", "blue cylinder", "brown cylinder", "cyan cube", "gray cone", "green cone", "green cube", "purple cube", "red cone", "red cube", "yellow cylinder"]
    candidate_captions = [f"a photo of a {cls}" for cls in classes]
    print(candidate_captions)
    correct = []
    #define our target classificaitons, you can should experiment with these strings of text as you see fit, though, make sure they are in the same order as your class names above
    text = clip.tokenize(candidate_captions).to(device)

    rows_list = []

    for cls in class_names:
        print(cls)
        class_correct = []
        test_imgs = glob.glob(img_path + '/' + cls + '/*.png')
        for img in test_imgs:
            #print("here")
            #print(img)
            image = transform(Image.open(img)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                pred = classes[argmax(list(probs)[0])]

                dict1 = {'true': cls, 'pred': pred}
                rows_list.append(dict1)
                if pred == cls:
                    correct.append(1)
                    class_correct.append(1)
                else:
                    correct.append(0)
                    class_correct.append(0)
                    print(img)


        print('accuracy on class ' + cls + ' is :' + str(sum(class_correct)/len(class_correct)))
    print('accuracy on all is : ' + str(sum(correct)/len(correct)))

    df = pd.DataFrame(rows_list)
    df.to_csv(output_file)

def predict_two(img_path, output_file, model_path, prompt_path):
    if not os.path.exists(output_file):
        os.mkdir(output_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)
    if model_path != None:
        model.load_state_dict(torch.load(model_path))

    correct = []

    for clsname in os.listdir(img_path):
        rows_list = []
        print(clsname)
        #print(cls)
        prompt_file_name = clsname
        prompts = pd.read_csv(prompt_path+'/'+prompt_file_name+'.csv')
        class_names = prompts['class_name'].tolist()
        #print(class_names)
        candidate_captions = [f"a photo of a {clsn}" for clsn in class_names]
        #print(candidate_captions)
        text = clip.tokenize(candidate_captions).to(device)
        class_correct = []
        clas = clsname.split("_")
        cls = clas[0] + " " + clas[1]
        test_imgs = glob.glob(img_path+'/'+clsname+'/' + cls + '/*.png')
        for img in test_imgs:
            #print(img)
            print("here")
            image = transform(Image.open(img)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                pred = class_names[argmax(list(probs)[0])]

                print(pred)
                print(cls)

                dict1 = {'true': cls, 'pred': pred}
                rows_list.append(dict1)
                #print(pred)
                if pred == cls:
                    correct.append(1)
                    class_correct.append(1)
                else:
                    correct.append(0)
                    class_correct.append(0)

        print('accuracy on class ' + clsname + ' is :' + str(sum(class_correct)/len(class_correct)))
        df = pd.DataFrame(rows_list)
        df.to_csv(output_file +'/'+ prompt_file_name + '.csv')
    print('accuracy on all is : ' + str(sum(correct)/len(correct)))

def predict_rel(img_path, output_file, model_path, prompt_path):
    save_dir = output_file
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)
    if model_path != None:
        model.load_state_dict(torch.load(model_path))

    correct = []

    for clsname in os.listdir(img_path):
        rows_list = []
        print(clsname)
        #print(cls)
        prompt_file_name = clsname
        #prompt_file_name = prompt_file_name.replace(" ", "_")
        prompts = pd.read_csv(prompt_path+'/'+prompt_file_name+'.csv')
        class_names = prompts['class_name'].tolist()
        for c in class_names:
            print("'" + c + "'")
        #print(class_names)
        candidate_captions = [f"a photo of a {clsn.split()[0]} to the {clsn.split()[1]} of a {clsn.split()[2]}" for clsn in class_names]
        #print(candidate_captions)
        text = clip.tokenize(candidate_captions).to(device)
        class_correct = []
        # clas = clsname.split("_")
        # cls = clas[0] + " " + clas[1]
        clas = clsname.replace("_"," ")
        test_imgs = glob.glob(img_path+'/'+clsname+'/'+clas+'/*.png')
        for img in test_imgs:
            #print(img)
            #print("here")
            image = transform(Image.open(img)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                pred = class_names[argmax(list(probs)[0])]

                print(pred)
                print(clas)

                dict1 = {'true': clas, 'pred': pred}
                print(dict1)
                rows_list.append(dict1)
                #print(pred)
                if pred == clas:
                    correct.append(1)
                    class_correct.append(1)
                else:
                    correct.append(0)
                    class_correct.append(0)

        print('accuracy on class ' + clsname + ' is :' + str(sum(class_correct)/len(class_correct)))
        df = pd.DataFrame(rows_list)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(save_dir +'/' + prompt_file_name+'_output.csv')
    print('accuracy on all is : ' + str(sum(correct)/len(correct)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='Path to folder of images to evalutate')
    parser.add_argument('--dataset', required=True, help='Single, Two_object or Relational')
    parser.add_argument('--output_file', required=True, help='Name of output csv file')
    parser.add_argument('--model_path', default=None, help='Path to fine-tuned CLIP model weights') 
    parser.add_argument('--prompt_path', default=None, help='Required for two object and relational')
    args = parser.parse_args()
    if args.dataset.lower() == 'single':
        predict(img_path=args.image_folder, output_file=args.output_file, model_path=args.model_path)
    elif args.dataset.lower() == 'two_object':
        predict_two(img_path=args.image_folder, output_file=args.output_file, model_path=args.model_path, prompt_path=args.prompt_path)
    elif args.dataset.lower()=='relational':
        predict_rel(img_path=args.image_folder, output_file=args.output_file, model_path=args.model_path, prompt_path=args.prompt_path)