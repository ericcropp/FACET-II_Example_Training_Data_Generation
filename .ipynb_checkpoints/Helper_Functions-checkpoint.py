from impact import Impact
import numpy as np
from scipy import stats
import time

import scipy.optimize 
import numpy as np
from scipy.stats import rv_continuous, norm, beta, gamma
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import json
from PIL import Image,ImageFilter
import pandas as pd
plt.rcParams.update({'font.size': 14})#`ArithmeticError
from xopt import Xopt
import xopt
from impact import Impact
import impact
from concurrent.futures import ProcessPoolExecutor
import os
from distgen import Generator
import cv2
import copy
import fastkde
import math
from pmd_beamphysics import ParticleGroup

from pmd_beamphysics.units import (nice_array, nice_scale_prefix,
                                   plottable_array)


               
####### Make Synthetic Distribution ########
def Gaussian_Dist_Maker(n,mu,sigma,lSig,rSig):
    """
    This function returns a truncated gaussian distribution of quasi-random particles.  This uses the Halton series
    
    Argument:
    n -- int number of particles
    mu -- float: center of distribution/mean
    sigma -- float: std of distribution
    lSig -- float number of sigma at which to truncate Gaussian left
    rSig -- float number of sigma at which to truncate Gaussian right
    """
    # Check inputs
    try: n = int(n)
    except: raise ValueError("n is not an int!")
    
    try: mu = float(mu)
    except: raise ValueError("mu is not a float!")
    
    try: sigma = float(sigma)
    except: raise ValueError("sigma is not a float!")
    
    try: lSig = float(lSig)
    except: raise ValueError("lSig is not a float!")
    
    try: rSig = float(rSig)
    except: raise ValueError("rSig is not a float!")
    
    
    # get and shuffle n samples from halton series
    h=scipy.stats.qmc.Halton(1)
    X0=h.random(n=n)
    np.random.shuffle(X0)
    
    # Make these into Gaussian and return
    X0=X0*(1-(1-scipy.stats.norm.cdf(lSig))-(1-scipy.stats.norm.cdf(rSig)))
    X0=X0+(1-scipy.stats.norm.cdf(lSig))
    GaussDist = mu + np.sqrt(2)*sigma*scipy.special.erfinv(2*X0-1)
    return np.squeeze(GaussDist)

def subfolder_number(folder):
    """
    This function determines and returns the maximum of the numbered folders in a directory.  Useful for determining the directory for the output.
    
    Argument:
    folder -- string of the folder in which to search
    
    """
    
    def int_filename(filename):
        """
        Returns the integer number of a filename, -1 if the folder is not a numbered folder.
        
        """
        try: 
            f=int(filename)
            return f
        except: 
            return -1


    subfolders = [ int_filename(f.name) for f in os.scandir(folder) if f.is_dir() ]
    
    try:
        num=max(subfolders)
    except:
        num=-1
    
    return num

def update_settings(new_settings,old_settings_dict):
    """
    This function appends the constant settings in the dictionary to the dataframe of variable settings.  
    Returns the appended Dataframe
    
    Argument:
    new_settings: pd.DataFrame of settings
    old_settings_dict: dictionary of constant settings
    """
    try:
        new_settings = pd.DataFrame(new_settings)
    except:
        raise ValueError("Problem with new_settings -- cannot convert to pd.DataFrame!")
    assert type(old_settings_dict) is dict, "old_settings_dict is not a dictionary!"
    
    old_settings=pd.DataFrame(old_settings_dict,index=[0])
    old_settings=pd.concat([old_settings] * len(new_settings)).sort_index().reset_index(drop=True)
    for col in new_settings.columns:
        old_settings[col]=new_settings[col].values

        
    return old_settings

def update_distgen(G,settings=None,verbose=False):
    """
    This function updates the distgen object given a dictionary of settings.
    Returns an updated distgen object
    
    Argument:
    G: distgen object to be updated of settings
    settings: dictionary of settings
    verbose: bool of whether to print messages
    """
    assert type(settings) is dict, "settings variable is not a dict"
    G.verbose=verbose
    if settings:
        for key in settings:
            val = settings[key]
            if key.startswith('distgen:'):
                key = key[len('distgen:'):]
                if verbose:
                    print(f'Setting distgen {key} = {val}')
                G[key] = val
            
    
    return G

def update_impact(I,settings=None,
               verbose=False):
    """
    This function updates the Impact object given a dictionary of settings.
    Returns an updated Impact object
    
    Argument:
    I: Impact object to be updated of settings
    settings: dictionary of settings
    verbose: bool of whether to print messages
    """  
    assert type(settings) is dict, "settings variable is not a dict"
    I.verbose=verbose
    if settings is not None:
        for key in settings:
            val = settings[key]
            if not key.startswith('distgen:'):
               # Assume impact
                if verbose:
                    print(f'Setting impact {key} = {val}')          
                I[key] = val                
   
    return I

def split_impact_bmad_settings(settings):
    """
    This function splits impact and bmad settings from a combined dictionary.
    Returns imapact and bmad settings dictionaries.
    Relies on special Impact settings starting with "impact_"
    and Bmad settings starting with "Bmad_"
    
    Argument:
    settings: dictionary or pd.DataFrame of settings with above specification.
    """
    impact_settings = {}
    bmad_settings = {}
    for key in settings.keys():
        if 'impact_' in key: 
            # print("key: "+key)
            new_key = key.split('impact_')
            new_key = new_key[-1]
            # print("new_key: " + new_key)
            impact_settings[new_key] = settings[key]
        elif 'bmad_' in key:
            new_key = key.split('bmad_')
            new_key = new_key[-1]
            bmad_settings[new_key] = settings[key]

    return impact_settings,bmad_settings

def combine_impact_bmad_settings(impact_settings,bmad_settings):
    """
    Combine impact and bmad settings into one pd.DataFrame of settings. 
    This reverses the above function.  Combined settings will start with "impact_" or 
    "Bmad_" to indicate which simulation code they belong to.
    
    Argument:
    impact_settings: pd.DataFrame of impact settings
    bmad_settings: pd.DataFrame of Bmad settings
    
    """
    impact_cols = impact_settings.columns
    bmad_cols = bmad_settings.columns
    
    new_cols = []
    for col in impact_cols:
        new_cols.append('impact_' + col)
    impact_settings.columns = new_cols
    
    new_cols = []
    for col in bmad_cols:
        new_cols.append('bmad_' + col)
    bmad_settings.columns = new_cols
    
    settings = pd.concat([impact_settings,bmad_settings],axis=1)
    return settings

def run_autophase(nominal_settings1,Impact_yml):
    """
    This function runs the autophase and scale for Impact using the nominal settings and 
    Impact Yaml file.
    
    Argument:
    nominal_settings1: dictionary of nominal settings
    Impact_yml: string indicating location of Impact Yaml input file
    """
    nominal_settings ={}
    for key in nominal_settings1.keys():
        if 'distgen' not in key and 'group' not in key:
            # print(key)
            nominal_settings[key] = nominal_settings1[key]
    I = impact.Impact.from_yaml(Impact_yml)
    I = update_impact(I,nominal_settings)
    I2=copy.deepcopy(I)

    P0 = pmd_beamphysics.single_particle(pz=1e-15, z=1e-15)


    I2.numprocs=1
    t=I2.track(P0,s=0.9)

    E=t['mean_energy']
    # print(E)
    target_L0AF=E-nominal_settings1['L0AF_scale:rf_field_scale']
    # print('target: ' +str(target_L0AF))
    phs,amp = autophase_and_scale(I,phase_ele_name='L0AF_phase', scale_ele_name='L0AF_scale', target=target_L0AF, scale_range=(10e6, 100e6), initial_particles=P0, verbose=False)
    return phs,amp

def run_impact_bmad(settings,impact_config=None,
                    distgen_input_file=None,
                    bmad_loc=None,
                    end_name='PR10571',
                    verbose=False):
    """
    This function runs impact and Bmad and returns a dictionary of where to find the output files.  Note that there is a special convention for the inputs.
    Settings must be specified as a dictionary, where settings to be passed to Impact must have a key starting with "impact_" and settings to be passed to 
    Bmad must start with "Bmad_".  There is one other key that must be present: "filepath" specifies the filepath in which to archive and run the simulations.  
    
    Argument:
    settings: dictionary as written above
    impact_config: string of path to impact input yaml file
    distgen_input_file: string of path to distgen input yaml file. 
    verbose: bool to indicate outputs
    """
    output = {}
    def update_bmad_settings(bmad_settings,tao,L0energy):
        """
        This function does not return, but updates the bmad settings
        
        """
        # Functions to scan L1 and L2 phases
        def getEnergyChangeFromElements(activeMatchStrings,tao):
            #VOLTAGE is just gradient times length; need to manually include phase info
            voltagesArr = [tao.lat_list(i, "ele.VOLTAGE", flags="-no_slaves -array_out") for i in activeMatchStrings]
            voltagesArr = np.concatenate(voltagesArr)

            angleMultArr = [tao.lat_list(i, "ele.PHI0", flags="-no_slaves -array_out") for i in activeMatchStrings]
            angleMultArr = np.concatenate(angleMultArr)
            angleMultArr = [np.cos(i * (2*3.1415) ) for i in angleMultArr] #Recall Bmad uses units of 1/2pi

            return( np.dot(voltagesArr, angleMultArr) )
        def setLinacGradientAuto(activeMatchStrings, targetVoltage,tao): 
            #Set to a fixed gradient so everything is the same. Not exactly physical but probably close
            baseGradient = 1.0e7
            tao=tao
            for i in activeMatchStrings: tao.cmd(f'set ele {i} GRADIENT = {baseGradient}')

            #See the resulting voltage gain
            voltageSum  = getEnergyChangeFromElements(activeMatchStrings,tao)

            #print(voltageSum/1e6)

            #Uniformly scale gradient to hit target voltage
            for i in activeMatchStrings: tao.cmd(f'set ele {i} GRADIENT = {baseGradient*targetVoltage/voltageSum}')

            voltageSum  = getEnergyChangeFromElements(activeMatchStrings,tao)

            #print(voltageSum/1e6)

        def setLinacPhase(activeMatchStrings, phi0Deg,tao):
            for i in activeMatchStrings: tao.cmd(f'set ele {i} PHI0 = {phi0Deg / 360.}')
        for key in bmad_settings.keys():
            if 'theta0_deg' in key:
                temp_name = key.split('_')[0]
                tao.cmd('set ele {} PHI0={}'.format(temp_name, (bmad_settings[key])/360))
            elif 'rf_field_scale' in key:
                temp_name = key.split('_')[0]
                tao.cmd('set ele {} VOLTAGE={}'.format(temp_name, bmad_settings[key]))
            elif 'Q' in key:
                tao.cmd('set ele {} B1_GRADIENT={}'.format(key, (bmad_settings[key])))
                
            if key=='L1phase':
                setLinacPhase(L1MatchStrings, bmad_settings[key],tao)
            if key=='L1energy':
                setLinacGradientAuto( L1MatchStrings, bmad_settings[key] - L0energy,tao )
            if key=='L2phase':
                setLinacPhase(L2MatchStrings, bmad_settings[key],tao)
            if key=='L2energy':
                try:
                    L1energy = bmad_settings['L1energy']
                except:
                    L1energy = 0.338e9 # Default value
                setLinacGradientAuto( L2MatchStrings, bmad_settings[key] - L1energy,tao )
                
                
                
    # Split impact and bmad settings
    impact_settings,bmad_settings = split_impact_bmad_settings(settings)
    
    # Get Impact simulation ID
    i = int(impact_settings["impactT_ID"])
    del impact_settings["impactT_ID"]
    
    # Set up the Impact and Distgen objects
    I = Impact.from_yaml(impact_config)
    G = Generator(distgen_input_file)
    
    # Update the Impact settings
    for key in impact_settings.keys():
        try:
            if float(impact_settings[key]) % 1 == 0:
                impact_settings[key] = int(float(impact_settings[key]))
            else:
                impact_settings[key] = float(impact_settings[key])
        except: pass
    
    I = update_impact(I,impact_settings)
        
    # Make and insert distribution
    G=update_distgen(G,impact_settings,verbose=False)
    G.input
    G.run()
    P = G.particles
    I.initial_particles = P
    
    # Get the proper filepath and working directory
    
    wd = settings['filepath'] +str(i)+'/'
    archive_path = settings['filepath'] +str(i)+'/'
    path = Path(wd)
    path.mkdir(parents=True, exist_ok=True)
    I.workdir = wd
    
    # Run impact
    I.run()
    
    # Archive 
    fingerprint = impact.impact_distgen.fingerprint_impact_with_distgen(I, G)
    path.mkdir(parents=True, exist_ok=True)
    archive_file = os.path.join(archive_path, fingerprint+'.h5')
    output['impact_archive'] = archive_file
    I.archive(archive_file)
    
    # Prepare particles (use openPMD) 
    fingerprint = impact.impact_distgen.fingerprint_impact_with_distgen(I, G)
    bfile = os.path.join(archive_path, fingerprint+'_bmad_init'+'.h5')
    P1 = I.particles['L0AFEND'].copy()
    # P1 = P1.resample(100_000)
    P1.drift_to_z()
    P1.write(bfile)
    output['bmad_init'] = bfile
    OpenPMD_to_Bmad(bfile,tOffset=None)
    
   
    # set bmad initial params, including stop and quads and inital distribution
    os.environ['OMP_NUM_THREADS']=str(1) # Important! Make sure this executes one evaluation on each core
    tao=Tao('-init {:s}/bmad/models/f2_elec/tao.init -noplot'.format(environ['FACET2_LATTICE'])) 
    
    # Set up Downstream
    # Find element strings matching L1, L2 and L3
    #These more general match strings might cause problems by including both lords and slaves!
    L1MatchStrings = ["K11_1*", "K11_2*"]
    L2MatchStrings = ["K11_4*", "K11_5*", "K11_6*", "K11_7*", "K11_8*", "K12_*", "K13_*", "K14_*"]
    L3MatchStrings = ["K15_*", "K16_*", "K17_*", "K18_*", "K19_*"]
    
    #Therefore, use lat_list and these general search strings to get specific elements
    #Recall that, by default, lat_list has the flag "-track_only" so lords are not included
    #But some of the tracking elements are super_slaves which cannot be `set`
    
    #"-array_out" flag is on by default but if flags are edited, need to re-add manually to get numeric values back
    L1MatchStrings = np.concatenate( [ tao.lat_list(i,"ele.name", flags="-no_slaves") for i in L1MatchStrings ] )
    L2MatchStrings = np.concatenate( [ tao.lat_list(i,"ele.name", flags="-no_slaves") for i in L2MatchStrings ] )
    L3MatchStrings = np.concatenate( [ tao.lat_list(i,"ele.name", flags="-no_slaves") for i in L3MatchStrings ] )
    
    L0energy = P1['mean_kinetic_energy']
    update_bmad_settings(bmad_settings,tao,L0energy)
    tao.cmd('reinit beam')
    tao.cmd('set beam_init track_end='+str(end_name))
    tao.cmd('set beam_init position_file='+bfile)
    tao.cmd('call Activate_CSR.tao');
    Bmad_CSR_Command = 'csroff' #or csron
    tao.cmd(Bmad_CSR_Command)
    
    # Run bmad
    tao.cmd('set global track_type = beam') 
    
    # archive bmad
    bmad_out_file = archive_path + fingerprint+'_bmad_end.h5'
    tao.cmd('write beam -at '+ end_name+ ' ' + bmad_out_file)
    output['bmad_out'] = bmad_out_file
    
    # Return bmad archive locations
    return output
    

def Linac_Calc(P_T):
    """
    This function uses the equations and variables from the SLAC Blue Book to calculate the 
    energy gain of the beam (neglecting beam loading) using the measured power in the waveguide.
    
    Reference: https://www.slac.stanford.edu/spires/hep/HEPPDF/twomile/Stanford_Two_Mile_Complete.pdf
    See page 116 and equation 6-7
    """

    Q = 13e3
    w = 2.856e9 *2*np.pi
    tF = 0.83e-6
    tau = w*tF/(2*Q)
    # P_T = 35.5e6
    L = 4.04879-1.058398
    r_0 = 53e6

    V_T = (1-np.exp(-2*tau))**0.5*(P_T*L*r_0)**0.5

    return V_T


class renamer():
    """
    This allows for dataframes' columns to be renamed if duplicated
    # From https://stackoverflow.com/questions/40774787/renaming-columns-in-a-pandas-dataframe-with-duplicate-column-names/40774999#40774999
    """
    def __init__(self):
          self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])