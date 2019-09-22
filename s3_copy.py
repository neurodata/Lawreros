#%%
# Set up boto3 and ask and specify bucket/paths
import boto3

session = boto3.Session(aws_access_key_id='', aws_secret_access_key='')
s3 = boto3.resource('s3')

bucket = s3.Bucket('ndmg-data')
original_dir = 'SWU4'
new_dir = 'SWU4-KeyError'

#%% Search and output dict with all file paths
for obj in bucket.objects.filter(Prefix=f'{original_dir}/sub'):
    print('{0}:{1}'.format(bucket.name,obj.key))


#%% Copy files from dict to the new directory
for sub in x:
    for ses in x[sub]:
        for obj in bucket.objects.filter(Prefix=f'{original_dir}/sub-'+sub+'/ses-'+ses):
            copy_source = {'Bucket':'ndmg-data',
            'Key':obj.key}
            s3.Object('ndmg-data',f'{new_dir}/{obj.key.split("/")[1]}/{obj.key.split("/")[2]}/{obj.key.split("/")[3]}/{obj.key.split("/")[4]}').copy(copy_source)

#%% dict containing all subjects and sessions you want to move
x = {'0025848': ['1', '2'],
 '0025777': ['1', '2'],
 '0025666': ['1', '2'],
 '0025839': ['1', '2'],
 '0025684': ['1', '2'],
 '0025635': ['1', '2'],
 '0025818': ['1', '2'],
 '0025746': ['1', '2'],
 '0025661': ['1', '2'],
 '0025658': ['1', '2'],
 '0025675': ['1', '2'],
 '0025629': ['1'],
 '0025636': ['2'],
 '0025638': ['1'],
 '0025640': ['2'],
 '0025650': ['2'],
 '0025669': ['1'],
 '0025674': ['2'],
 '0025676': ['2'],
 '0025677': ['2'],
 '0025693': ['1'],
 '0025694': ['1'],
 '0025703': ['1'],
 '0025704': ['1'],
 '0025709': ['1'],
 '0025712': ['1'],
 '0025720': ['1'],
 '0025723': ['2'],
 '0025726': ['1'],
 '0025727': ['2'],
 '0025733': ['1'],
 '0025734': ['1'],
 '0025739': ['2'],
 '0025741': ['2'],
 '0025749': ['1'],
 '0025756': ['1'],
 '0025767': ['1'],
 '0025769': ['1'],
 '0025776': ['2'],
 '0025779': ['2'],
 '0025780': ['2'],
 '0025782': ['1'],
 '0025786': ['2'],
 '0025794': ['1'],
 '0025796': ['1'],
 '0025805': ['1'],
 '0025806': ['2'],
 '0025812': ['1'],
 '0025825': ['1'],
 '0025828': ['1'],
 '0025834': ['1'],
 '0025835': ['1'],
 '0025840': ['2'],
 '0025842': ['2'],
 '0025845': ['1'],
 '0025849': ['1'],
 '0025856': ['1'],
 '0025862': ['1']}

#%%
