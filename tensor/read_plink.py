"""Read plink files by predefined chuncks."""
from pyplink import PyPlink
import pandas as pd
import os
from google.cloud import storage
import pickle
import numpy as np
from pytorch_regression import pytorch_linear
from sklearn.preprocessing import scale


class genetic_testdata(object):
    """Import and process genetic data in plink format for ML."""

    def __init__(self, download_path):
        """Perpare genetic data."""
        super(genetic_testdata, self).__init__()
        self.download_path = download_path
        self._gc_client = storage.Client('Hail')
        self._bucket = self._gc_client.get_bucket('ukb_testdata')

    def _download(self, gc_path, output_path):
        if ('gs' in gc_path) or ('ukb_testdata' in gc_path):
            raise ValueError('path is the path AFTER the bucket name')
        blob = self._bucket.blob(gc_path)
        blob.download_to_filename(output_path)

    def download_1kg_chr22(self):
        """Download chr22 from the 1K Genome."""
        files = ['1kg_phase1_chr22.' + k for k in ['bed', 'bim', 'fam']]
        print('start downloading files')
        for f in files:
            output_path = os.path.join(self.download_path, f)
            gc_path = os.path.join('data', f)
            if os.path.isfile(output_path):
                continue
            self._download(gc_path, output_path)
        print('Files were donwloaded to {}'.format(self.download_path))
        return os.path.join(self.download_path, '1kg_phase1_chr22')

    def download_ukb_chr10(self):
        """Download chr10 from the UKB (maf>=0.01)."""
        files = ['maf_0.01_10.' + k for k in ['bed', 'bim', 'fam']]
        print('start downloading files')
        for f in files:
            output_path = os.path.join(self.download_path, f)
            gc_path = os.path.join('data', f)
            if os.path.isfile(output_path):
                continue
            self._download(gc_path, output_path)
        print('Files were donwloaded to {}'.format(self.download_path))
        return os.path.join(self.download_path, 'maf_0.01_10')

    def download_ldblocks(self):
        """Download LD block file."""
        file = 'Berisa.EUR.hg19.bed'
        output_path = os.path.join(self.download_path, file)
        gc_path = os.path.join('data', file)
        if not os.path.isfile(output_path):
            self._download(gc_path, output_path)
        return os.path.join(self.download_path, file)

    def download_file(self, file):
        """Download any given file."""
        output_path = os.path.join(self.download_path, os.path.basename(file))
        gc_path = file
        if not os.path.isfile(output_path):
            self._download(gc_path, output_path)
        return output_path


class DataProcessing(object):
    """Processe phenotypes."""

    def __init__(self, pheno):
        """Process phenotypes."""
        super(DataProcessing, self).__init__()
        assert isinstance(pheno, str)
        assert os.path.isfile(pheno)
        self.pheno_path = pheno

    def get_pheno(self, x):
        """Get a specific phenotype."""
        assert x > 0
        assert isinstance(x, int)
        smaller_pheno_file = os.path.join("/tmp",
                                          'smaller_pheno.tab')
        os.system('cut -f1,2,3,4,5,{} {} > {}'.format(str(x+6),
                                                      self.pheno_path,
                                                      smaller_pheno_file))

        pheno = pd.read_csv(smaller_pheno_file, sep='\t')
        pheno = pheno.rename(columns={pheno.columns.values[-1]: 'pheno'})
        return pheno


class Genetic_data_read(object):
    """Provides batch reading of plink Files."""

    def __init__(self, plink_file, batches=None):
        """Provide batch reading of plink Files."""
        super(Genetic_data_read, self).__init__()
        self.plink_file = plink_file
        self.plink_reader = PyPlink(plink_file)
        self.bim = self.plink_reader.get_bim()
        self.n = self.plink_reader.get_nb_samples()
        self.bim.columns = [k.strip() for k in self.bim.columns]
        self.chromosoms = self.bim.chrom.unique()
        if batches is not None:
            self._dirname = os.path.dirname(plink_file)
            self._pickel_path = plink_file+'.ld_blocks.pickel'
            if os.path.isfile(self._pickel_path):
                self.groups = pickle.load(open(self._pickel_path, 'rb'))
            else:
                self.groups = pd.read_csv(batches, sep='\t')
                self.groups.columns = [k.strip() for k in self.groups.columns]
                self.groups['chr'] = self.groups['chr'].str.strip('chr')
                self.groups['chr'] = self.groups['chr'].astype(int)
                self.groups = self._preprocessing_ldblock()
                pickle.dump(self.groups, open(self._pickel_path, 'wb'))
        else:
            self.groups = None

    def _preprocessing_ldblock(self):
        if self.groups is None:
            return None
        else:
            out = {}
            for chr in self.chromosoms:
                subset_blocks = self.groups[self.groups.chr == chr]
                subset_bim = self.bim[self.bim.chrom == chr]
                out[chr] = []
                print(subset_bim.dtypes)
                print(subset_blocks.dtypes)
                for index, row in subset_blocks.iterrows():
                    start = row['start']
                    end = row['stop']
                    rsids = subset_bim[
                        (self.bim.pos >= start)
                        & (self.bim.pos <= end)
                         ].index.values
                    out[chr].append(rsids)
            return out

    def _get_block(self, snp, chr):
        block_id = None
        pos_id = None
        len_block = None
        for i, block in enumerate(self.groups[chr]):
            len_block = len(block)
            for u, id in enumerate(block):
                if snp == id:
                    block_id = i
                    pos_id = u
                    return block_id, pos_id, len_block

    def block_iter(self, chr=22):
        """Block iteration."""
        assert chr in self.chromosoms
        current_block = 0
        block_ids = self.groups[chr][current_block]
        size_block = len(block_ids)
        genotypematrix = np.zeros((self.n, size_block))
        pos_id = 0
        for snp, genotypes in self.plink_reader.iter_geno():
            if snp not in block_ids:
                continue
            else:
                genotypematrix[:, pos_id] = genotypes
                pos_id += 1
                if pos_id >= (size_block - 1):
                    yield genotypematrix
                    pos_id = 0
                    current_block += 1
                    block_ids = self.groups[chr][current_block]
                    size_block = len(block_ids)
                    genotypematrix = np.zeros((self.n, size_block))


if __name__ == '__main__':
    download_path = 'tensor/data/'
    sim_path = 'data/phenotypes/simulated_chr10.txt'
    downloader = genetic_testdata(download_path)
    # plink_stem = downloader.download_1kg_chr22()
    plink_stem = downloader.download_ukb_chr10()
    ld_blocks = downloader.download_ldblocks()
    pheno_file = downloader.download_file(sim_path)
    pheno_reader = DataProcessing(pheno_file)
    ph = pheno_reader.get_pheno(1)
    y = ph['pheno'].values

    genetic_process = Genetic_data_read(plink_stem, ld_blocks)
    out = genetic_process.block_iter(10)
    X = next(out)
    X = scale(X)
    model_comparision_file = os.path.join(download_path, 'model.comparisions')
    pytorchmodel = pytorch_linear(X, y, model_comparision_file,
                                  False, type='c')
    pytorchmodel.run(penal='l1')
