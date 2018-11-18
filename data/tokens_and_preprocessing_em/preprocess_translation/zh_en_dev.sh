DATAPATH=/Users/mansimov/Downloads/zh-en

src=zh
trg=en
lang=zh-en

echo "pre-processing valid/test data..."
for l in $src $tgt; do
  o=$DATAPATH/IWSLT15.TED.tst2013.$lang.$l.xml
  fname=${o##*/}
  f=/tmp/${fname%.*}
  echo $o $f
  grep '<seg id' $o | \
      sed -e 's/<seg id="[0-9]*">\s*//g' | \
      sed -e 's/\s*<\/seg>\s*//g' | \
      sed -e "s/\’/\'/g" > $f
  echo ""
done
