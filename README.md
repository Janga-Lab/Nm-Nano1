# Nm-Nano
Nm-Nano: A framework for predicting 2´-O-Methylation (Nm) Sites in Nanopore RNA Sequencing Data

# Getting Started and pre-requisites
The following softwares and modules should be installed before using  Nm-Nano

python 3.6.10

minimpa2 (https://github.com/lh3/minimap2)

Nanopolish (https://github.com/jts/nanopolish)

samtools (http://www.htslib.org/)

numpy 1.18.1

pandas 1.0.1

sklearn 0.22.2.post1

tensorflow 2.0.0

keras 2.3.1 (using Tensorflow backend)


# Running  Nm-Nano:

In order to run  Nm-Nano, the user has do the following:

1- Ensure that BED file that highlights the Nm modified locations on the whole genome is in the same path where  main.py file exists:
2- Run the following python command:

python main.py -r ref.fa -f reads.fastq

Where the  Nm-Nano framework needs the following two inputs files when running it:

- A reference Genome file (ref.fa)
- The fastq reads file (reads.fastq)

# Note:
- The user should enter the BED file name with the absolute path and extension 

- The user should include the fast5 files folder (fast5_files) from which reads.fastq file was generated in the same path of main.py

- The default model used in Nm-Nano framework is the Xgboost model, but the user can test the Random Forest (RF) with embedding model by simply editing the last line  in main.py file to refer to RF with embedding model implemented in RF_embedding_test_split.py instead of xgboost implemented in xgboost_test_split.py
