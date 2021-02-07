import psycopg2
import pandas as pd
# code based on https://github.com/allenai/pubmedextract/blob/master/analysis_scripts/04_analysis.py

# diseases_query = '''
# select c.nct_id, c.mesh_term as mesh
# from browse_conditions as c join studies as d on c.nct_id = d.nct_id
# where d.overall_status = 'Completed';
# '''
# diseases = pd.read_sql_query(diseases_query, conn)

sex_query = '''
select s.study_first_submitted_date, b.nct_id, b.category as cat, sum(b.param_value_num) as total_participants
from
(
    select bm.nct_id, lower(concat(bm.category, bm.classification)) as category, bm.param_value_num, bm.title, bm.param_type
    from baseline_measurements as bm join result_groups as r
    on bm.nct_id = r.nct_id and bm.ctgov_group_code = r.ctgov_group_code
    and lower(r.title) != 'total'
) as b
join studies as s on b.nct_id = s.nct_id
where s.overall_status = 'Completed' 
and s.study_first_submitted_date < '2021-01-01'
and (b.param_type = 'Count of Participants' or b.param_type = 'Number')
and (b.title ~* 'sex' or b.title ~* 'gender') 
and (b.category = 'male' or b.category = 'female')
group by b.nct_id, cat, s.study_first_submitted_date
order by b.nct_id;
'''


def execute_query():
    params = {'dbname': 'aact', 'host': 'aact-db.ctti-clinicaltrials.org', 'port': 5432,
              'user': input("AACT username:"), 'password': input("AACT password:")}
    conn = psycopg2.connect(**params)
    return pd.read_sql_query(sex_query, conn)


if __name__ == "__main__":
    sex = execute_query()
    # keep nct_ids that have both male and female
    fields = ['nct_id', 'study_first_submitted_date', 'total_participants']
    df = sex[sex['cat'] == 'female'][fields].merge(sex[sex['cat'] == 'male'][fields], on='nct_id',
                                                   suffixes=['_f', '_m'])
    df = df.drop(columns=['study_first_submitted_date_f'], axis=1)
    df = df.rename({'total_participants_f': 'female',
                    'total_participants_m': 'male',
                    'study_first_submitted_date_m': 'first_submitted_date'}, axis=1)
    df.dropna(axis=0, how='any', inplace=True)
    df.to_csv('ncts_with_participants.csv', index=False)
