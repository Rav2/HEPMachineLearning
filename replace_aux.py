def read_json(from_where):
    f = open(from_where,'r')
    return json.loads(f.read())