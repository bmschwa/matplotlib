import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import tempfile
#from matplotlib.testing.decorators import knownfailureif
from nose.tools import raises
import numpy.testing as nptest

def test_rec_query_config_construct():
    x = mlab.RecQueryConfig()
    x1 = mlab.RecQueryConfig(tablename='atable')
    x3 = mlab.RecQueryConfig(tablename='tbl')
    x4 = mlab.RecQueryConfig(tablename='tbl')

def test_duplicate_tables():
    config = mlab.RecQueryConfig(tablename='testTable')
    rin = np.array([(1.0, 2), (3.0, 4), (1.5, 55.0)], dtype=[('x', float), ('y', int)])

    mlab.rec2sql(rin, config)
    rout_2 = mlab.sql2rec(config, 'select count(*) as myCount from testTable;')
    assert( len(rin) == rout_2[0][0] )
    
    mlab.rec2sql(rin, config)
    rout_3 = mlab.sql2rec(config, 'select count(*) as myCount from testTable;')
    assert( rout_2[0][0] + len(rin) == rout_3[0][0] )

    rout_4 = mlab.rec_query(rin, 'select count(*) as myCount from testTable;', config)
    assert( rout_3[0][0] + len(rin) == rout_4[0][0] )

    rout_5 = mlab.rec_query(rin, 'select count(*) as myCount from testTable;', config)
    assert( rout_4[0][0] + len(rin) == rout_5[0][0] )


def test_table_counts():
    config = mlab.RecQueryConfig(tablename='testTable')
    #rin = np.core.records.array([(99,6.0,3.14),(-99,-6.0,-3.14),(0.0,0,0.0)], names=('x','y','t'))
    rin = np.core.records.array([('str','10/9/1978','true'),('another string', '10-9-1978', 'false')], names=('x','y','z'))
    len1 = len(rin)
    
    mlab.rec2sql(rin, config)
    rout_2 = mlab.sql2rec(config)
    _compare_helper(rin, rout_2) #we're converting the dates.  that's good
    assert( len1 == len(rout_2) )

def _compare_converted_dates(xin, xout):
    try:     
        import dateutil.parser
        import datetime      
        tIn = map( dateutil.parser.parse, xin)
        tOut = map( dateutil.parser.parse, map(str,xout) )

        return np.array_equal(tIn, tOut )
    except Exception:
        return False
    assert( False )

def _compare_converted_bools(xin, xout):
    try:
        tIn = map( str.lower, map(str,xin) )                
        tOut = map( str.lower, map(str,xout) )
        return tIn == tOut
    except Exception:
        return False
    assert( False )
    
def _compare_helper(rin, rout):
    assert( len(rin.dtype.names) == len(rout.dtype.names) )
    err = False
    for n in rin.dtype.names:
        if( ( rin[n].dtype != rout[n].dtype ) ):
            #before we call this an error let's make sure we didn't convert
            if ( _compare_converted_dates(rin[n], rout[n]) ):
                pass
            elif( _compare_converted_bools(rin[n], rout[n]) ):
                pass     
            else:
                err = True
        if( cbook.is_numlike( rin[n] ) and cbook.is_numlike( rout[n] ) ):
            if( np.allclose(rin[n], rout[n]) == False):
                err = True
        elif( cbook.is_string_like( rin[n] ) and cbook.is_string_like( rout[n] ) ):
            if( np.array_equal(rin[n], rout[n] ) == False): #can't do this for everything b/c of dates
                err = True
        if err:    
            print 'In: ', rin[n]
            print 'Out: ', rout[n]
            raise ValueError('sent %s got %s in %s' % (rin[n].dtype, rout[n].dtype, n) )


def test_round_trip():
    from datetime import date
    import datetime
    x = np.core.records.array([(1, 1.2, date.today(),datetime.datetime.utcnow()), \
                                (2,2.1,date(1941,12,7), datetime.datetime.utcnow())], \
                                names=('Id', 'aFloat', 'ADate', 'MyTime') )

    f = mlab.rec2sql(x)
    config = mlab.RecQueryConfig(tablename=f['tablename'], database=f['database'])
    x_from_func = mlab.sql2rec(config)
    _compare_helper(x, x_from_func)

    datafile = cbook.get_sample_data('goog.npy')
    r = np.load(datafile).view(np.recarray)
    config.tablename = 'new_tablexxx'    
    r_from_func = mlab.rec_query(r, 'Select * from new_tablexxx', config)
    _compare_helper(r, r_from_func)

    r_from_func2 = mlab.sql2rec(config)
    _compare_helper(r_from_func, r_from_func2)

    s = np.array([(1.0, 2), (3.0, 4), (1.5, 55.0)], dtype=[('x', float), ('y', int)])
    config.tablename='s_test' 
    mlab.rec2sql(s, config)
    s_out = mlab.sql2rec(config)    
    _compare_helper(r_from_func, r_from_func2)
"""
Demonstrate how DB Functionality works with svn revisions in the data.

    svn co https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/sample_data

and edit testdata.csv to add a new row.  After committing the changes,
when you rerun this script you will get the updated data (and the new
svn version will be cached in ~/.matplotlib/sample_data)
"""


import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import matplotlib as mpl
import numpy as np

def _compare_helper(rin, rout):
    assert( len(rin.dtype.names) == len(rout.dtype.names) )
    for n in rin.dtype.names:
        if( rin[n].dtype != rout[n].dtype ):
            print 'sent %s got %s in %s' % (rin[n].dtype, rout[n].dtype, n)
            print '%s ?= %s --> %s' % (rin[n], rout[n], rin[n] == rout[n] )


datafile = cbook.get_sample_data('goog.npy')
r = np.load(datafile).view(np.recarray)

pnl = np.zeros_like(r.adj_close)
pnl[1:] = np.diff(r.adj_close) * 100
pnl[0] = (r.adj_close[0] - r.open[0]) * 100
years = [row['date'].year for row in r]
months = [row['date'].month for row in r]
r = mlab.rec_append_fields(r, ('year', 'month', 'pnl'),  (years, months, pnl))

assert( pnl.size == len(years) == len(months) )

config = mlab.RecQueryConfig(tablename='tbl',close_conn_when_done=False)

test1 = mlab.rec_query(r, 'Select count(*) from tbl', config)
print test1

test2 = mlab.rec_query(None, 'Select * from tbl', config)
assert( len(test2) == len(r) )
_compare_helper( r, test2 )

test3 = mlab.rec_query(None, 'Select year,month,avg(pnl) as a from tbl group by year, month', config)

t4 = mlab.rec_query(None, 'Select month, year, pnl from tbl', config)
config.tablename='new_table'
print t4
t5 = mlab.rec2sql(t4, config)
_compare_helper( t4, t5 )
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import tempfile
#from matplotlib.testing.decorators import knownfailureif
from nose.tools import raises

def test_rec_query_config_construct():
    x = mlab.RecQueryConfig()
    x1 = mlab.RecQueryConfig(tablename='atable')
    x3 = mlab.RecQueryConfig(tablename='tbl')
    x4 = mlab.RecQueryConfig(tablename='tbl')

def test_duplicate_tables():
    config = mlab.RecQueryConfig(tablename='testTable')
    rin = np.array([(1.0, 2), (3.0, 4), (1.5, 55.0)], dtype=[('x', float), ('y', int)])
    rout_1 = mlab.rec2sql(rin, config)
    firstCount = len(rout_1)
    rout_2 = mlab.rec2sql(rin, config)
    assert( len(rout_2) == firstCount*2 )

def test_table_counts():
    config = mlab.RecQueryConfig(tablename='testTable')
    #rin = np.core.records.array([(99,6.0,3.14),(-99,-6.0,-3.14),(0.0,0,0.0)], names=('x','y','t'))
    rin = np.core.records.array([('str','10/9/1978','true'),('another string', '10-9-1978', 'false')], names=('x','y','z'))
    rout_1 = mlab.rec2sql(rin, config)
    firstCount = len(rout_1)
    rout_2 = mlab.rec2sql(rin, config)
    #_compare_helper(rin, rout_1) #we're converting the dates.  that's good
    assert( len(rout_2) == firstCount*2 )

def _compare_helper(rin, rout):
    assert( len(rin.dtype.names) == len(rout.dtype.names) )
    for n in rin.dtype.names:
        if( rin[n].dtype != rout[n].dtype ):
            raise ValueError('sent %s got %s in %s' % (rin[n].dtype, rout[n].dtype, n) )

    def what_type(self):
        import types
        return types.FloatType

def test_data_types():
    from datetime import date
    import datetime
    x = np.core.records.array([(1, 1.2, date.today(),datetime.datetime.utcnow()), \
                                (2,2.1,date(1941,12,7), datetime.datetime.utcnow())], \
                                names=('Id', 'aFloat', 'ADate', 'MyTime') )

    x_from_func = mlab.rec2sql(x)
    _compare_helper(x, x_from_func)

    #datafile = cbook.get_sample_data('goog.npy')
    #r = np.load(datafile).view(np.recarray)
    #r_from_func = mlab.rec2sql(r)
    #_compare_helper(r, r_from_func)
