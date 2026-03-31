#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/stoker

mkdir -p $data/stoker/raw

curl -O  https://www.gutenberg.org/cache/epub/345/pg345.txt
mv pg345.txt $data/stoker/raw/dracula.txt

# preprocess slightly

cat $data/stoker/raw/dracula.txt | python $base/scripts/preprocess_raw.py > $data/stoker/raw/dracula.cleaned.txt

# tokenize, fix vocabulary upper bound

#cat $data/stoker/raw/dracula.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > $data/stoker/raw/dracula.preprocessed.txt
python $base/scripts/preprocess.py \
  --vocab-size 5000 \
  --tokenize \
  --lang en \
  --sent-tokenize \
< $data/stoker/raw/dracula.cleaned.txt \
> $data/stoker/raw/dracula.preprocessed.txt

# split into train, valid and test

head -n 440 $data/stoker/raw/dracula.preprocessed.txt | tail -n 400 > $data/stoker/valid.txt
head -n 840 $data/stoker/raw/dracula.preprocessed.txt | tail -n 400 > $data/stoker/test.txt
tail -n 3075 $data/stoker/raw/dracula.preprocessed.txt | head -n 2955 > $data/stoker/train.txt
