import os
import os.path as p
import subprocess
from time import time
from datetime import date, timedelta

from dt_sim.data_reader.date_sort_funcs import pub_date_split


def main(input_file: str, date_dir: str, daily_dir: str):

    #### Init paths and prepare ingestion file
    t_0 = time()

    # Get file ISO date
    yyyy, mm, dd = str(input_file.split('/')[-1]).split('.')[0].split('-')

    print('\n\nChecking paths... \n\n')
    date_split_dir = p.join(date_dir, f'{yyyy}_extraction_{mm}-{dd}')
    daily_idx_dir = p.join(daily_dir, f'{yyyy}_indexes_{mm}-{dd}')
    # mkdir pub_date_split/YYYY_extraction_MM-DD/ --> Crash if exists
    # mkdir faiss_index_shards/deployment_full/daily_in/YYYY_indexes_MM-DD/ --> Crash if exists
    try:
        os.mkdir(date_split_dir)
        os.mkdir(daily_idx_dir)
    except FileExistsError:
        print(f'{input_file} has already been processed \n')
        return 0

    t_split = time()
    print(f'Splitting {input_file} by publication date... ')
    # Date split --> to_dir=pub_date_split/YYYY_extraction_MM-DD/*.jl
    # cutoff=(45 days before ingest)
    last_45_days = date.isoformat(date(int(yyyy), int(mm), int(dd)) - timedelta(45))
    pub_date_split(input_file, output_dir=date_split_dir,
                   cutoff_date=f'{last_45_days}', ingest_date=f'{yyyy}-{mm}-{dd}')

    msp, ssp = divmod(time()-t_split, 60)
    print(f'Splitting completed in {int(msp)}m{ssp:0.2f}s \n\n')

    # For .jl in date_split_dir...
    split_news = list()
    for (parent, _, jls) in os.walk(date_split_dir):
        for jl in jls:
            if jl.endswith('.jl'):
                split_news.append(p.join(parent, jl))
        break

    #### Vectorize & index
    print(f'Vectorizing {len(split_news)} days of news... \n\n')
    vectorize = '/faiss/dig-text-similarity-search' \
                '/py_scripts/preprocessing/prep_shard.py'
    for _ in range(len(split_news)):
        os.system(f'python -u {vectorize} {date_split_dir} {daily_idx_dir} '
                  f'-p {p.join(date_split_dir, "progress.txt")} '
                  f'-b /faiss/dig-text-similarity-search/'
                  f'base_indexes/USE_large_base_IVF4K_15M.index '
                  f'-l -n 128 -v -t 2; ')

    # Report time spent vectorizing
    m, s = divmod(time()-t_0, 60)
    print(f'\n\nVectorized {input_file} '
          f'into {len(split_news)} shards '
          f'in {int(m)}m{s:0.2f}s \n\n')

    #### Save backup
    file_handler = '/faiss/dig-text-similarity-search/' \
                   'py_scripts/preprocessing/consolidate_shards.py'
    backup_dir = f'/faiss/faiss_index_shards/backups/WL_{mm}{dd}'
    os.system(f'python {file_handler} {daily_idx_dir} {backup_dir} --cp;')

    #### Switch to tmp service (daemon)
    service_script = '/faiss/dig-text-similarity-search' \
                     '/py_scripts/service/similarity_server.py'
    tmp_indexes = '/green_room/idx_deploy_B'
    tmp_log_file = f'/faiss/dig-text-similarity-search/logs/service/tmp_deploy_{mm}{dd}.out'
    # Use os.system() to launch service & continue ingestion
    os.system('kill -15 $(ps -ef | grep "[s]imilarity_server" | awk \'{print $2}\'); '
              'sleep 1;')
    os.system(f'python {service_script} {tmp_indexes} -l -c 6 & ')

    #### Indexes before zip_merge
    deployment_full = '/faiss/faiss_index_shards/deployment_full/'
    shards_before = list()
    for (parent, _, idx_files) in os.walk(deployment_full):
        for idxf in idx_files:
            if any(ext in idxf for ext in ['.index', '.ivfdata']):
                shards_before.append(p.join(parent, idxf))
        break

    #### Zip ingest into deployment_full (do not delete daily indexes)
    pin = f'zip_to_{mm}{dd}'
    os.system(f'echo "n" | python {file_handler} {daily_idx_dir} {deployment_full} '
              f'--zip -p {pin} -t 2;')

    #### Indexes after merge
    shards_after = list()
    for (parent, _, idx_files) in os.walk(deployment_full):
        for idxf in idx_files:
            if any(ext in idxf for ext in ['.index', '.ivfdata']):
                shards_after.append(p.join(parent, idxf))
        break

    #### Relaunch main service (daemon)
    log_file = f'/faiss/dig-text-similarity-search/logs/service/deploy_{mm}{dd}.out'
    os.system('kill -15 $(ps -ef | grep "[s]imilarity_server" | awk \'{print $2}\'); '
              'sleep 1;')
    os.system(f'python -u {service_script} {deployment_full} -l -c 6 >> {log_file} & ')

    #### Zip ingest into tmp_indexes (delete daily indexes)
    os.system(f'echo "y" | python {file_handler} {daily_idx_dir} {tmp_indexes} '
              f'--zip -p {pin} -t 2;')

    #### Cleanup
    # rm temp files -- leave YYYY_extraction_MM-DD/progress.txt
    os.system(f'rm {p.join(date_split_dir, "*.jl")}; '
              f'rm {p.join(date_split_dir, "*/*.jl")}; '
              f'rmdir {p.join(date_split_dir, "old_news")}; '
              f'rmdir {p.join(date_split_dir, "date_error")};')

    # Move input_file to OUT dir & gzip
    os.rename(input_file, input_file.replace("_IN/", "_OUT/"))
    os.system(f'gzip {input_file.replace("_IN/", "_OUT/")} &')

    #### Backup to S3
    old_indexes = sorted([idxpth for idxpth in shards_before if idxpth not in shards_after])
    new_indexes = sorted([idxpth for idxpth in shards_after if idxpth not in shards_before])

    s3rm = '/faiss/dig-text-similarity-search/s3rm.sh'
    for idx in old_indexes:
        subprocess.run([f'{s3rm}', f'{idx}'])

    s3cp = '/faiss/dig-text-similarity-search/s3cp.sh'
    for idx in new_indexes:
        subprocess.run([f'{s3cp}', f'{idx}'])

    # Report time spent overall
    m_fin, s_fin = divmod(time()-t_0, 60)
    print(f'\n\n\nIngested {input_file} in {int(m_fin)}m{s_fin:0.2f}s \n\n')

    return 1


if __name__ == '__main__':
    # Static dir paths
    news_dir = '/faiss/sage_news_data/raw/LN_WLFilt_extractions_IN/'
    pub_date_dir = '/faiss/sage_news_data/pub_date_split/'
    daily_ingest_dir = '/faiss/faiss_index_shards/tmp_daily_ingest/'

    # Find the file in sage_news_data/raw/LN_WLFilt_extractions_IN/
    candidates = list()
    for (parent_dir, _, files) in os.walk(news_dir):
        for f in files:
            if any(ext in f for ext in ['.jl', '.v2', '.gz']):
                candidates.append(p.join(parent_dir, f))
        break
    candidates.sort()

    if len(candidates):
        # if file.jl.gz --> gunzip
        target_file = p.abspath(candidates[0])
        print(f'Processing {target_file}')
        if target_file.endswith('.gz'):
            os.system(f'gunzip {target_file};')
        target_file = target_file.replace('.gz', '')

        main(input_file=target_file, date_dir=pub_date_dir, daily_dir=daily_ingest_dir)
    else:
        print('No files to process \n')
