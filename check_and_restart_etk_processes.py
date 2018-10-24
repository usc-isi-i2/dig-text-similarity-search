import requests
import json
from argparse import ArgumentParser


def check_and_restart_etk_processes(mydig_url, project_name, number_of_workers):
    etk_status_url = '{}/dig_etl_engine/etk_status/{}'.format(mydig_url, project_name)
    etk_status_response = requests.get(etk_status_url)
    if etk_status_response.status_code // 100 == 2:
        # so far so good, lets see if the etk processes are running
        etk_processes = json.loads(etk_status_response.content)['etk_processes']
        if etk_processes == 0:
            # etk down! time to revive
            run_etk_url = '{}/dig_etl_engine/run_etk'.format(mydig_url)
            params = dict()
            params['project_name'] = project_name
            params['number_of_workers'] = number_of_workers
            response = requests.post(run_etk_url, data=json.dumps(params))
            return response.text
        return 'etk processes already running'
    return 'can not check if etk processes are up and running'


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-u", "--mydigurl", action="store", type=str, dest="mydigurl")
    parser.add_argument("-p", "--project", action="store", type=str, dest="project_name")
    parser.add_argument("-n", "--workers", action="store", type=int, dest="workers", default=8)

    args, _ = parser.parse_known_args()
    print(check_and_restart_etk_processes(args.mydigurl, args.project_name, args.workers))