# Exécuter dans 'defi-2025-transformers'
# Installe torch en local, puis fine-tune BERT (Transformers env sans torch)
import subprocess, sys
# Installer torch localement si absent
try:
    import torch
except ImportError:
    print("Installation locale de torch...")
    subprocess.check_call([sys.executable,'-m','pip','install','--user','torch'])
    import torch

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Config
model_name='bert-base-uncased'; epochs=3; bs=16; max_len=128

# Chargement CSV nettoyés
df=pd.read_csv('train_cleaned.csv')
labels=pd.read_csv('train_label.csv')
df=df.merge(labels,on='Id')
df['Clean']=df['Clean'].fillna("")
# Label encoding
lbl2id={l:i for i,l in enumerate(sorted(df['Category'].unique()))}
id2lbl={i:l for l,i in lbl2id.items()}
df['lbl_id']=df['Category'].map(lbl2id)
# Split
tr,va=train_test_split(df,test_size=0.1,stratify=df['lbl_id'],random_state=42)
# Tokenizer+model
tok=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=len(lbl2id),id2label=id2lbl,label2id=lbl2id)

def compute(p):
    preds=np.argmax(p.predictions,axis=1); labs=p.label_ids
    P,R,F,_=precision_recall_fscore_support(labs,preds,average='weighted')
    return {'accuracy':accuracy_score(labs,preds),'f1':F,'precision':P,'recall':R}

class DS(torch.utils.data.Dataset):
    def __init__(self, txt, lbl=None): self.txt, self.lbl=txt,lbl
    def __len__(self): return len(self.txt)
    def __getitem__(self,i):
        enc=tok(self.txt[i],truncation=True,padding='max_length',max_length=max_len,return_tensors='pt')
        itm={k:v.squeeze(0) for k,v in enc.items()}
        if self.lbl is not None: itm['labels']=torch.tensor(self.lbl[i])
        return itm

# Datasets
dtr=DS(tr['Clean'].tolist(),tr['lbl_id'].tolist()); dva=DS(va['Clean'].tolist(),va['lbl_id'].tolist())
# Trainer
args=TrainingArguments(output_dir='./out',num_train_epochs=epochs,per_device_train_batch_size=bs,per_device_eval_batch_size=bs,evaluation_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='accuracy')
trainer=Trainer(model,args,train_dataset=dtr,eval_dataset=dva,compute_metrics=compute)
# Train
trainer.train()

# Prédiction
tdf=pd.read_csv('test_cleaned.csv'); tds=DS(tdf['Clean'].tolist())
preds=trainer.predict(tds).predictions
ids=np.argmax(preds,axis=1)
# Soumission output
tmpl=pd.read_csv('template_submissions.csv')
tmpl['Category']=[id2lbl[i] for i in ids]
tmpl.to_csv('submission_bert.csv',index=False)
print("Submission générée.")
