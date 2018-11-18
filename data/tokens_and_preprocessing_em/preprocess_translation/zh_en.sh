DATAPATH=/Users/mansimov/Downloads/zh-en

src=zh
trg=en
lang=zh-en

echo "pre-processing train data..."
for l in $src $trg; do
  f=train.tags.$lang.$l
  tok=train.tags.$lang.tok.$l
  cat $DATAPATH/$f | \
  grep -v '<url>' | \
  grep -v '<url' | \
  grep -v '<talkid>' | \
  grep -v '<talkid' | \
  grep -v '<keywords>' | \
  grep -v '<keywords' | \
  grep -v '<speaker>' | \
  grep -v '<speaker' | \
  grep -v '<reviewer>' | \
  grep -v '<reviewer' | \
  grep -v '<translator>' | \
  grep -v '<translator' | \
  sed -e 's/<title>//g' | \
  sed -e 's/<\/title>//g' | \
  sed -e 's/<description>//g' | \
  sed -e 's/<\/description>//g' > /tmp/$tok
  echo ""
done
