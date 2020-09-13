python setup.py install --record files.txt
cat files.txt | xargs /bin/rm -rf
/bin/rm -rf build/
/bin/rm -rf dist/
pip uninstall hpneat -y
/bin/rm files.txt
