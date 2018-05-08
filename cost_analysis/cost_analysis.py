"""Collection of functions to get a better idea of cost."""
from google.cloud import bigquery
import pandas as pd


class CloudCost(object):
    """Handy class to check our cloud cost."""
    def __init__(self, project_id="hail-201606"):
        self.project_id = project_id
        self.database = "`hail-201606.billing.gcp_billing_export_v1_00F470_51FAFA_0E175E`"
        self.bigquery_client = bigquery.Client(self.project_id)
        self.location = 'asia-northeast1'

    def cost_label(self, label, verbose=False):
        """Query the total cost of a given label."""
        query = (
                'SELECT * FROM %s, UNNEST(labels) AS l '
                'WHERE l.value="%s"' % (self.database, label))
        print(query)
        query_job = self.bigquery_client.query(query, location=self.location)
        overall_cost = 0
        for row in query_job:
            overall_cost += row['cost']
        if verbose:
            print(label+":", round(overall_cost, 2), "USD")
        return(overall_cost)

    def to_pandas(self):
        """Load the current cost table into a pandas dataframe."""
        query = (
            'SELECT * from %s' % self.database
        )
        query_job = self.bigquery_client.query(query, location=self.location)
        rows = list(query_job.result(timeout=30))

        out = []
        for row in query_job:
            r = {}
            for key, item in row.items():
                if isinstance(item, dict):
                    process = item
                    for k, i in process.items():
                        name = key+"_"+k
                        r[name] = i
                elif isinstance(item, list):
                    if not len(item) > 0:
                        continue
                    process = item[0]
                    assert(isinstance(process, dict))
                    for k, i in process.items():
                        name = key+"_"+k
                        r[name] = i
                else:
                    r[key] = item
            out.append(r)
        return(pd.DataFrame(out))
