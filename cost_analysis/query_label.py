"""Collection of functions to get a better idea of cost."""
from google.cloud import bigquery
bigquery_client = bigquery.Client('hail-201606')


def estimate_cost(label):
    """Query cost of a given label."""
    database = \
        "`hail-201606.billing.gcp_billing_export_v1_00F470_51FAFA_0E175E`"
    query = (
        'SELECT * FROM %s, UNNEST(labels) AS l '
        'WHERE l.value="%s"' % (database, label))
    query_job = bigquery_client.query(query, location='asia-northeast1')
    overall_cost = 0
    for row in query_job:
        overall_cost += row['cost']
    print(label+":", round(overall_cost, 2), "USD")
    return(overall_cost)
