for folder in */; do
  cd $folder
  for file in *.ipynb; do
    jupyter nbconvert --execute --ExecutePreprocessor.enabled=True \
      --ExecutePreprocessor.timeout=600 --to notebook --inplace $file
    jupyter nbconvert --execute --ExecutePreprocessor.enabled=True \
      --ExecutePreprocessor.timeout=600 --to markdown $file
  done
  cd -
done
