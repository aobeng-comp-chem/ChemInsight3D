
# This function is based on the real spherical harmonics to cartesian GTO conversion
# Adapted from IODATA transformation from pure to Cartesian transformation without normalization.
# where the original expressions for the angular part of the wavefunctions were detailed.
# The conversion of these expressions into Python functions was performed to facilitate
# computational analysis within this codebase.



def ang_res_lamda(dx, dy, dz, orb_val):

    """
    Calculate the angular part of a wavefunction for a given orbital.

    This function computes the angular part of the wavefunction for an electron in a specified orbital.
    The calculation is based on the real spherical harmonics and depends on the differential coordinates
    (dx, dy, dz) of the electron relative to the nucleus.

    Parameters:
    dx (float): The x-component of the differential position vector.
    dy (float): The y-component of the differential position vector.
    dz (float): The z-component of the differential position vector.
    orb_val (str): A string representing the type of orbital ('s', 'px', 'py', 'pz' up to J functions).

    Returns:
    float: The value of the angular part of the wavefunction if orb_val is valid; otherwise, None.

    The function uses a dictionary to map the 'orb_val' parameter to the corresponding mathematical function
    representing the angular part of the wavefunction. If 'orb_val' is not recognized, the function returns None.

    Example:
    >>> ang_res_lamda(1, 0, 0, 'px')
    1
    >>> ang_res_lamda(0, 1, 0, 'py')
    1
    >>> ang_res_lamda(0, 0, 1, 'pz')
    1
    >>> ang_res_lamda(1, 1, 1, 'd0')
    -0.5
    """

    
    
    # Define a dictionary to map orb_val to its corresponding function
    angular_part_dict = {
        's': lambda dx, dy, dz: 1,
        'px': lambda dx, dy, dz: dx,
        'py': lambda dx, dy, dz: dy,
        'pz': lambda dx, dy, dz: dz,
        
        'd0': lambda dx, dy, dz: 0.5 * (2*dz*dz - dx*dx - dy*dy),
        'dc1': lambda dx, dy, dz: 1.7320508075688773 * dy * dz,
        'ds1': lambda dx, dy, dz: 1.7320508075688773 * dx * dz,
        'dc2': lambda dx, dy, dz: 0.86602540378443865 * (dx*dx - dy*dy),
        'ds2': lambda dx, dy, dz: 1.7320508075688773 * dx * dy,
        
        'f0': lambda dx, dy, dz: -1.5 * dx*dx*dz - 1.5 * dy*dy*dz + dz*dz*dz,
        'fc1': lambda dx, dy, dz: -0.61237243569579452 * dx*dx*dx - 0.61237243569579452 * dx*dy*dy + 2.4494897427831781 * dx*dz*dz,        
        'fs1': lambda dx, dy, dz: -0.61237243569579452 * dx*dx*dy - 0.61237243569579452 * dy*dy*dy + 2.4494897427831781 * dy*dz*dz,
        'fc2': lambda dx, dy, dz:  1.9364916731037084 * dx*dx*dz - 1.9364916731037084 * dy*dy*dz,
        'fs2': lambda dx, dy, dz: 3.8729833462074169 * dx*dy*dz,
        'fc3': lambda dx, dy, dz: 0.79056941504209483 * dx*dx*dx - 2.3717082451262845 * dx*dy*dy,
        'fs3': lambda dx, dy, dz: 2.3717082451262845 * dx*dx*dy - 0.79056941504209483 * dy*dy*dy,
        
        'g0':  lambda dx, dy, dz: 0.375*dx**4 + 0.75*dx**2*dy**2 - 3.0*dx**2*dz**2 + 0.375*dy**4 - 3.0*dy**2*dz**2 + 1.0*dz**4,
        'gc1': lambda dx, dy, dz: -2.37170824512628*dx**3*dz - 2.37170824512628*dx*dy**2*dz + 3.16227766016838*dx*dz**3    ,
        'gs1': lambda dx, dy, dz: -2.37170824512628*dx**2*dy*dz - 2.37170824512628*dy**3*dz + 3.16227766016838*dy*dz**3,
        'gc2': lambda dx, dy, dz: -0.559016994374947*dx**4 + 3.35410196624968*dx**2*dz**2 + 0.559016994374947*dy**4 - 3.35410196624968*dy**2*dz**2,
        'gs2': lambda dx, dy, dz: -1.11803398874989*dx**3*dy - 1.11803398874989*dx*dy**3 + 6.70820393249937*dx*dy*dz**2,
        'gc3': lambda dx, dy, dz: 2.09165006633519*dx**3*dz - 6.27495019900557*dx*dy**2*dz,
        'gs3': lambda dx, dy, dz: 6.27495019900557*dx**2*dy*dz - 2.09165006633519*dy**3*dz,
        'gc4': lambda dx, dy, dz: 0.739509972887452*dx**4 - 4.43705983732471*dx**2*dy**2 + 0.739509972887452*dy**4,
        'gs4': lambda dx, dy, dz: 2.95803989154981*dx**3*dy - 2.95803989154981*dx*dy**3,
        
        'h0': lambda dx, dy, dz: 1.875*dx**4*dz + 3.75*dx**2*dy**2*dz - 5.0*dx**2*dz**3 + 1.875*dy**4*dz - 5.0*dy**2*dz**3 + 1.0*dz**5,
        'hc1': lambda dx, dy, dz:  0.484122918275927*dx**5 + 0.968245836551854*dx**3*dy**2 - 5.80947501931113*dx**3*dz**2 + 0.484122918275927*dx*dy**4 - 5.80947501931113*dx*dy**2*dz**2 + 3.87298334620742*dx*dz**4,
        'hs1': lambda dx, dy, dz: 0.484122918275927*dx**4*dy + 0.968245836551854*dx**2*dy**3 - 5.80947501931113*dx**2*dy*dz**2 + 0.484122918275927*dy**5 - 5.80947501931113*dy**3*dz**2 + 3.87298334620742*dy*dz**4,
        'hc2': lambda dx, dy, dz: -2.5617376914899*dx**4*dz + 5.1234753829798*dx**2*dz**3 + 2.5617376914899*dy**4*dz - 5.1234753829798*dy**2*dz**3,
        'hs2': lambda dx, dy, dz:  -5.1234753829798*dx**3*dy*dz - 5.1234753829798*dx*dy**3*dz + 10.2469507659596*dx*dy*dz**3,
        'hc3': lambda dx, dy, dz: -0.522912516583797*dx**5 + 1.04582503316759*dx**3*dy**2 + 4.18330013267038*dx**3*dz**2 + 1.56873754975139*dx*dy**4 - 12.5499003980111*dx*dy**2*dz**2,
        'hs3': lambda dx, dy, dz: -1.56873754975139*dx**4*dy - 1.04582503316759*dx**2*dy**3 + 12.5499003980111*dx**2*dy*dz**2 + 0.522912516583797*dy**5 - 4.18330013267038*dy**3*dz**2,
        'hc4': lambda dx, dy, dz:  2.21852991866236*dx**4*dz - 13.3111795119741*dx**2*dy**2*dz + 2.21852991866236*dy**4*dz,
        'hs4': lambda dx, dy, dz: 8.87411967464942*dx**3*dy*dz - 8.87411967464942*dx*dy**3*dz,
        'hc5': lambda dx, dy, dz:  0.701560760020114*dx**5 - 7.01560760020114*dx**3*dy**2 + 3.50780380010057*dx*dy**4,
        'hs5': lambda dx, dy, dz:  3.50780380010057*dx**4*dy - 7.01560760020114*dx**2*dy**3 + 0.701560760020114*dy**5,
        
        'i0': lambda dx, dy, dz: -0.3125*dx**6 - 0.9375*dx**4*dy**2 + 5.625*dx**4*dz**2 - 0.9375*dx**2*dy**4 + 11.25*dx**2*dy**2*dz**2 - 7.5*dx**2*dz**4 - 0.3125*dy**6 + 5.625*dy**4*dz**2 - 7.5*dy**2*dz**4 + 1.0*dz**6,
        'ic1': lambda dx, dy, dz: 2.8641098093474*dx**5*dz + 5.7282196186948*dx**3*dy**2*dz - 11.4564392373896*dx**3*dz**3 + 2.8641098093474*dx*dy**4*dz - 11.4564392373896*dx*dy**2*dz**3 + 4.58257569495584*dx*dz**5,
        'is1': lambda dx, dy, dz: 2.8641098093474*dx**4*dy*dz + 5.7282196186948*dx**2*dy**3*dz - 11.4564392373896*dx**2*dy*dz**3 + 2.8641098093474*dy**5*dz - 11.4564392373896*dy**3*dz**3 + 4.58257569495584*dy*dz**5,
        'ic2': lambda dx, dy, dz: 0.45285552331842*dx**6 + 0.45285552331842*dx**4*dy**2 - 7.24568837309472*dx**4*dz**2 - 0.45285552331842*dx**2*dy**4 + 7.24568837309472*dx**2*dz**4 - 0.45285552331842*dy**6 + 7.24568837309472*dy**4*dz**2 - 7.24568837309472*dy**2*dz**4,
        'is2': lambda dx, dy, dz:  0.90571104663684*dx**5*dy + 1.81142209327368*dx**3*dy**3 - 14.4913767461894*dx**3*dy*dz**2 + 0.90571104663684*dx*dy**5 - 14.4913767461894*dx*dy**3*dz**2 + 14.4913767461894*dx*dy*dz**4,
        'ic3': lambda dx, dy, dz: -2.71713313991052*dx**5*dz + 5.43426627982104*dx**3*dy**2*dz + 7.24568837309472*dx**3*dz**3 + 8.15139941973156*dx*dy**4*dz - 21.7370651192842*dx*dy**2*dz**3,
        'is3': lambda dx, dy, dz: -8.15139941973156*dx**4*dy*dz - 5.43426627982104*dx**2*dy**3*dz + 21.7370651192842*dx**2*dy*dz**3 + 2.71713313991052*dy**5*dz - 7.24568837309472*dy**3*dz**3,
        'ic4': lambda dx, dy, dz:  -0.496078370824611*dx**6 + 2.48039185412305*dx**4*dy**2 + 4.96078370824611*dx**4*dz**2 + 2.48039185412305*dx**2*dy**4 - 29.7647022494766*dx**2*dy**2*dz**2 - 0.496078370824611*dy**6 + 4.96078370824611*dy**4*dz**2,
        'is4': lambda dx, dy, dz: -1.98431348329844*dx**5*dy + 19.8431348329844*dx**3*dy*dz**2 + 1.98431348329844*dx*dy**5 - 19.8431348329844*dx*dy**3*dz**2,
        'ic5': lambda dx, dy, dz:  2.32681380862329*dx**5*dz - 23.2681380862329*dx**3*dy**2*dz + 11.6340690431164*dx*dy**4*dz,
        'is5': lambda dx, dy, dz: 11.6340690431164*dx**4*dy*dz - 23.2681380862329*dx**2*dy**3*dz + 2.32681380862329*dy**5*dz,
        'ic6': lambda dx, dy, dz: 0.671693289381396*dx**6 - 10.0753993407209*dx**4*dy**2 + 10.0753993407209*dx**2*dy**4 - 0.671693289381396*dy**6,
        'is6': lambda dx, dy, dz:  4.03015973628838*dx**5*dy - 13.4338657876279*dx**3*dy**3 + 4.03015973628838*dx*dy**5,
        
        'j0': lambda dx, dy, dz:  -2.1875*dx**6*dz - 6.5625*dx**4*dy**2*dz + 13.125*dx**4*dz**3 - 6.5625*dx**2*dy**4*dz + 26.25*dx**2*dy**2*dz**3 - 10.5*dx**2*dz**5 - 2.1875*dy**6*dz + 13.125*dy**4*dz**3 - 10.5*dy**2*dz**5 + 1.0*dz**7,
        'jc1': lambda dx, dy, dz:  -0.413398642353842*dx**7 - 1.24019592706153*dx**5*dy**2 + 9.92156741649221*dx**5*dz**2 - 1.24019592706153*dx**3*dy**4 + 19.8431348329844*dx**3*dy**2*dz**2 - 19.8431348329844*dx**3*dz**4 - 0.413398642353842*dx*dy**6 + 9.92156741649221*dx*dy**4*dz**2 - 19.8431348329844*dx*dy**2*dz**4 + 5.29150262212918*dx*dz**6    ,
        'js1': lambda dx, dy, dz: -0.413398642353842*dx**6*dy - 1.24019592706153*dx**4*dy**3 + 9.92156741649221*dx**4*dy*dz**2 - 1.24019592706153*dx**2*dy**5 + 19.8431348329844*dx**2*dy**3*dz**2 - 19.8431348329844*dx**2*dy*dz**4 - 0.413398642353842*dy**7 + 9.92156741649221*dy**5*dz**2 - 19.8431348329844*dy**3*dz**4 + 5.29150262212918*dy*dz**6,
        'jc2': lambda dx, dy, dz:  3.03784720237868*dx**6*dz + 3.03784720237868*dx**4*dy**2*dz - 16.2018517460197*dx**4*dz**3 - 3.03784720237868*dx**2*dy**4*dz + 9.72111104761179*dx**2*dz**5 - 3.03784720237868*dy**6*dz + 16.2018517460197*dy**4*dz**3 - 9.72111104761179*dy**2*dz**5,
        'js2': lambda dx, dy, dz: 6.07569440475737*dx**5*dy*dz + 12.1513888095147*dx**3*dy**3*dz - 32.4037034920393*dx**3*dy*dz**3 + 6.07569440475737*dx*dy**5*dz - 32.4037034920393*dx*dy**3*dz**3 + 19.4422220952236*dx*dy*dz**5,
        'jc3': lambda dx, dy, dz: 0.42961647140211*dx**7 - 0.42961647140211*dx**5*dy**2 - 8.5923294280422*dx**5*dz**2 - 2.14808235701055*dx**3*dy**4 + 17.1846588560844*dx**3*dy**2*dz**2 + 11.4564392373896*dx**3*dz**4 - 1.28884941420633*dx*dy**6 + 25.7769882841266*dx*dy**4*dz**2 - 34.3693177121688*dx*dy**2*dz**4,
        'js3': lambda dx, dy, dz: 1.28884941420633*dx**6*dy + 2.14808235701055*dx**4*dy**3 - 25.7769882841266*dx**4*dy*dz**2 + 0.42961647140211*dx**2*dy**5 - 17.1846588560844*dx**2*dy**3*dz**2 + 34.3693177121688*dx**2*dy*dz**4 - 0.42961647140211*dy**7 + 8.5923294280422*dy**5*dz**2 - 11.4564392373896*dy**3*dz**4,
        'jc4': lambda dx, dy, dz:  -2.8497532787945*dx**6*dz + 14.2487663939725*dx**4*dy**2*dz + 9.49917759598167*dx**4*dz**3 + 14.2487663939725*dx**2*dy**4*dz - 56.99506557589*dx**2*dy**2*dz**3 - 2.8497532787945*dy**6*dz + 9.49917759598167*dy**4*dz**3,
        'js4': lambda dx, dy, dz: -11.399013115178*dx**5*dy*dz + 37.9967103839267*dx**3*dy*dz**3 + 11.399013115178*dx*dy**5*dz - 37.9967103839267*dx*dy**3*dz**3,
        'jc5': lambda dx, dy, dz: -0.474958879799083*dx**7 + 4.27462991819175*dx**5*dy**2 + 5.699506557589*dx**5*dz**2 + 2.37479439899542*dx**3*dy**4 - 56.99506557589*dx**3*dy**2*dz**2 - 2.37479439899542*dx*dy**6 + 28.497532787945*dx*dy**4*dz**2,
        'js5': lambda dx, dy, dz: -2.37479439899542*dx**6*dy + 2.37479439899542*dx**4*dy**3 + 28.497532787945*dx**4*dy*dz**2 + 4.27462991819175*dx**2*dy**5 - 56.99506557589*dx**2*dy**3*dz**2 - 0.474958879799083*dy**7 + 5.699506557589*dy**5*dz**2,
        'jc6': lambda dx, dy, dz:  2.4218245962497*dx**6*dz - 36.3273689437454*dx**4*dy**2*dz + 36.3273689437454*dx**2*dy**4*dz - 2.4218245962497*dy**6*dz,
        'js6': lambda dx, dy, dz: 14.5309475774982*dx**5*dy*dz - 48.4364919249939*dx**3*dy**3*dz + 14.5309475774982*dx*dy**5*dz,
        'jc7': lambda dx, dy, dz: 0.647259849287749*dx**7 - 13.5924568350427*dx**5*dy**2 + 22.6540947250712*dx**3*dy**4 - 4.53081894501425*dx*dy**6,
        'js7': lambda dx, dy, dz: 4.53081894501425*dx**6*dy - 22.6540947250712*dx**4*dy**3 + 13.5924568350427*dx**2*dy**5 - 0.647259849287749*dy**7 
   

    }

    # Use the get method of the dictionary to return the function for the given orb_val
    # If orb_val is not in the dictionary, it returns None
    func = angular_part_dict.get(orb_val)

    # Call the function with dx, dy, dz if it exists
    angular_part = func(dx, dy, dz) if func else None

    return angular_part
