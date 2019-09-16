from __future__ import absolute_import
import argparse
import logging
import re
from datetime import datetime
from past.builtins import unicode
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


class Split(beam.DoFn):
    def process(self, element):
        try:
            created_date, closed_date, incident_zip, location_type, community_board, agency, complaint_type = element.split(',')
            return [{
                'created_date': str(created_date),
                'closed_date': str(closed_date),
                'incident_zip': float(incident_zip),
                'location_type': str(location_type),
                'community_board': str(community_board),
                'agency': str(agency),
                'complaint_type': str(complaint_type)
            }]
        except:
            pass


class ChangeFormat(beam.DoFn):
    def process(self, element):
        try:
            element['created_date'] = element['created_date'][:-4]
            element['closed_date'] = element['closed_date'][:-4]
            return [element]
        except:
            pass
        


class RemoveData(beam.DoFn):
    def process(self, element):
        if(element['closed_date'] is not None):
            return [element]


class DayPeriod(beam.DoFn):
    def process(self, element):
        try:
            time = element['created_date']

            period = ''
            time = datetime.strptime(time[-8:], '%H:%M:%S')
            if((time >= datetime.strptime('06:00:00', '%H:%M:%S')) and (time < datetime.strptime('12:00:00', '%H:%M:%S'))):
                period = 'morning'
            elif((time >= datetime.strptime('12:00:00', '%H:%M:%S')) and (time < datetime.strptime('17:00:00', '%H:%M:%S'))):
                period = 'afternoon'
            elif((time >= datetime.strptime('17:00:00', '%H:%M:%S')) and (time < datetime.strptime('20:00:00', '%H:%M:%S'))):
                period = 'evening'
            else:
                period = 'night'

            element['day_period'] = period
            return [element]
        except:
            pass


class DayOfWeek(beam.DoFn):
    def process(self, element):
        try:
            time = element['created_date']
            weekday = ''
            t = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            day = t.weekday()
            if(day == 0 or day == 1):
                weekday = 'Mon-Tue'
            elif(day == 2 or day == 3):
                weekday = 'Wed-Thu'
            else:
                weekday = 'Fri-Sat-Sun'
            element['day_of_week'] = weekday
            return [element]
        except:
            pass


class TimeTaken(beam.DoFn):
    def process(self, element):
        try:
            created_date = str(element['created_date'])
            closed_date = str(element['closed_date'])
            to_ret = 0
            if(closed_date is not None and created_date is not None):
                t = datetime.strptime(closed_date, '%Y-%m-%d %H:%M:%S') - datetime.strptime(created_date, '%Y-%m-%d %H:%M:%S')
                if(t.days != 0):
                    to_ret += (t.days * 24 * 60)
                if(t.seconds != 0):
                    to_ret += (t.seconds / 60)
            element['TimeTaken'] = round(to_ret/60, 3)
            return [element]
        except:
            element['TimeTaken'] = None
            return [element]

"""
    python main_file.py \
    --runner DataflowRunner \
    --project summerai \
    --staging_location gs://nyc_servicerequest/staging \
    --temp_location gs://nyc_servicerequest/temp \
    --job_name pipeline-job \
    --input gs://nyc_servicerequest/Input/data000000000000.csv \
    --output gs://nyc_servicerequest/Output/data.csv\
    --setup_file setup.py
"""

PROJECT = 'summerai'
BUCKET = 'nyc_servicerequest'


def run(argv=None):
    argv = [
        '--project={0}'.format(PROJECT),
        '--job_name=pipeline-job',
        '--save_main_session',
        '--staging_location=gs://{0}/staging/'.format(BUCKET),
        '--temp_location=gs://{0}/staging/'.format(BUCKET),
        '--runner=DataflowRunner'
    ]

    # p = beam.Pipeline(argv=argv)
    input1 = 'gs://nyc_servicerequest/Input/data000000000000.csv'
    output_prefix = 'gs://nyc_servicerequest/Output/data.csv'

    """parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        dest='input',
                        default='gs://nyc_servicerequest/Input/data000000000000',
                        help='Input file to process.')
    parser.add_argument('--output',
                        dest='output',
                        # CHANGE 1/5: The Google Cloud Storage path is required
                        # for outputting the results.
                        default='gs://nyc_servicerequest/Output/data.csv',
                        help='Output file to write results to.')
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_args.extend([
        # CHANGE 2/5: (OPTIONAL) Change this to DataflowRunner to
        # run your pipeline on the Google Cloud Dataflow Service.
        '--runner=DirectRunner',
        # CHANGE 3/5: Your project ID is required in order to run your pipeline on
        # the Google Cloud Dataflow Service.
        '--project=summerai',
        # CHANGE 4/5: Your Google Cloud Storage path is required for staging local
        # files.
        '--staging_location=gs://nyc_servicerequest/staging',
        # CHANGE 5/5: Your Google Cloud Storage path is required for temporary
        # files.
        '--temp_location=gs://nyc_servicerequest/temp',
        '--job_name=pipeline-job',
    ])

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True"""

    # 1. Read the CSV File
    # 2. Change 'created_date' format
    # 3. Change 'closed_date' format
    # 4. Remove rows with 'closed_date' as null
    # 5. Add 'day_period' column
    # 6. Add 'day_of_week' column
    # 7. Compute 'TimeTaken'
    # 8. Select Columns and write to csv

    with beam.Pipeline(argv=argv) as p:
        # 1. Reading the csv and splitting lines by elements we want to retain
        #csv_lines = (p | "Read from CSV" >> beam.Create([input1]))
        csv_lines = (p | 'Read from CSV' >> beam.io.ReadFromText(input1, skip_header_lines=1) | beam.ParDo(Split()))

        # 2 and 3. Change 'created_date' format
        date_format = (csv_lines | 'Change Date Format' >> beam.ParDo(ChangeFormat()))

        # 4. Remove null 'closed_date entries'
        #remove_data = (date_format | 'Remove Null Entries' >> beam.ParDo(RemoveData()))

        # 5. Add 'day_period' column
        day_period = (date_format | 'Add Day Period' >> beam.ParDo(DayPeriod()))

        # 6. Add 'day_of_week' column
        day_of_week = (day_period | 'Add Day of Week' >> beam.ParDo(DayOfWeek()))

        # 7. Compute TimeTaken
        time_taken = (day_of_week | 'Add TimeTaken' >> beam.ParDo(TimeTaken()))

        # 8. Select Columns to write to csv
        write_vals = (time_taken | 'Read Values' >> beam.Map(lambda x: x.values()))
        write_csv = (write_vals | 'CSV Format' >> beam.Map(
            lambda row: ', '.join(['"'+str(column)+'"' for column in row])))
        #print(write_csv)
        write_csv | 'Write to GCS' >> beam.io.WriteToText(output_prefix,
            file_name_suffix='.csv',\
            header='created_date, closed_date, incident_zip, location_type, community_board, agency, complaint_type, day_period, day_of_week, TimeTaken')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
