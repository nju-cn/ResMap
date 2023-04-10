from matplotlib import pyplot as plt
import numpy as np
plt.rc('font',family='Times New Roman')
import matplotlib
matplotlib.rc('pdf', fonttype=42)
def auto_text(rects):
    for rect in rects:
        ax.text(rect.get_x()+0.0251, rect.get_height()+0.0516, rect.get_height(), ha='left',fontsize=14, va='bottom')

def auto_text1(rects):
    for rect in rects:
        ax.text(rect.get_x()+0.0, rect.get_height()+0.3516, rect.get_height(), ha='left',fontsize=14, va='bottom')
lbs_ifr_latency_pd_1=\
    [4.038853168487549, 3.915499210357666, 3.9784176349639893, 3.9105372428894043, 3.911146402359009, 4.00354528427124, 3.9953136444091797, 3.955702781677246, 3.940474271774292, 3.896298408508301, 3.9647629261016846, 3.9307947158813477, 3.977017641067505, 3.942357301712036, 3.965573310852051, 3.9145638942718506, 3.972477436065674, 3.947597026824951, 3.9493565559387207, 3.9368841648101807, 3.9525609016418457, 3.949044942855835, 3.873842477798462, 3.9004247188568115, 3.893258571624756, 3.9364824295043945, 3.9896206855773926, 3.9770877361297607, 3.9533491134643555, 4.11541748046875, 3.9168336391448975, 3.974416494369507, 3.922391176223755, 3.9639968872070312, 4.008809804916382, 3.954373359680176, 3.897251844406128, 3.9485607147216797, 3.9193639755249023, 3.9135167598724365, 3.9078242778778076, 3.9034321308135986, 4.005882501602173, 3.9968271255493164, 3.9666290283203125, 3.933432102203369, 3.870321273803711, 3.924372673034668, 3.9099206924438477, 4.0349438190460205]
acc_lbs_pd_1=[sum(lbs_ifr_latency_pd_1[:i+1]) for i in range(len(lbs_ifr_latency_pd_1))]

my_ifr_latency_pd_1=\
    [3.2890031337738037, 2.322774887084961, 2.2876741886138916, 2.3778181076049805, 2.442857265472412, 2.568530797958374, 3.2305335998535156, 3.254810333251953, 2.250483989715576, 2.2940149307250977, 3.1714096069335938, 2.2502353191375732, 3.2521023750305176, 2.3371658325195312, 2.379140853881836, 2.397677421569824, 2.5440096855163574, 2.3009426593780518, 2.295525550842285, 2.5102338790893555, 2.7575037479400635, 2.8617186546325684, 3.2815651893615723, 2.3163416385650635, 3.261460304260254, 2.3498499393463135, 2.2857556343078613, 2.3116295337677, 2.6096930503845215, 2.379009246826172, 2.343539237976074, 2.5624256134033203, 2.314434289932251, 2.2989981174468994, 2.603391170501709, 2.6115779876708984, 3.291877031326294, 2.217897653579712, 2.2906436920166016, 2.3127074241638184, 2.709601402282715, 2.3580732345581055, 2.3497276306152344, 3.148228168487549, 2.220062017440796, 2.392838954925537, 2.425808906555176, 2.3306241035461426, 3.2677056789398193, 2.2994582653045654]
acc_das_pd_1=[sum(my_ifr_latency_pd_1[:i+1]) for i in range(len(my_ifr_latency_pd_1))]

lbs_ifr_latency_pd_3=\
    [4.786261558532715, 5.8222815990448, 6.960336446762085, 4.856046199798584, 4.956571340560913, 4.990305423736572, 4.9083640575408936, 4.840731620788574, 4.874310255050659, 4.67769980430603, 4.6728222370147705, 4.750513076782227, 4.8280861377716064, 4.828605651855469, 4.710536479949951, 4.718026161193848, 4.756572484970093, 4.683918237686157, 4.99401593208313, 4.939972400665283, 4.9721949100494385, 4.856495380401611, 4.8132569789886475, 4.709017992019653, 4.876900911331177, 4.948244571685791, 5.056963205337524, 4.932087659835815, 4.887036561965942, 4.833484649658203, 4.967701435089111, 5.092220067977905, 5.065297842025757, 4.9483795166015625, 4.799597263336182, 4.900592803955078, 4.958624601364136, 4.947467803955078, 4.874657154083252, 4.834954500198364, 4.915255069732666, 5.127093553543091, 4.911654472351074, 4.836028575897217, 4.657893657684326, 4.959989070892334, 5.044149875640869, 4.986521244049072, 4.839910984039307, 4.717535018920898]
acc_lbs_pd_3=[sum(lbs_ifr_latency_pd_3[:i+1]) for i in range(len(lbs_ifr_latency_pd_3))]

my_ifr_latency_pd_3=\
    [4.05704140663147, 5.001888751983643, 6.261968374252319, 3.331404447555542, 3.7262532711029053, 3.4089043140411377, 3.7048656940460205, 3.8344154357910156, 4.0288920402526855, 3.836839199066162, 3.8541359901428223, 3.635004997253418, 3.843379259109497, 3.417623996734619, 3.7809698581695557, 3.240809202194214, 3.4873671531677246, 3.1345467567443848, 3.3549108505249023, 3.3741812705993652, 3.4264729022979736, 3.592939853668213, 4.100548028945923, 4.024808645248413, 3.9508814811706543, 3.1751320362091064, 3.4132282733917236, 3.581981897354126, 3.5983262062072754, 3.67535138130188, 3.3168692588806152, 3.450223207473755, 3.242100477218628, 3.4361042976379395, 3.413740396499634, 3.481414556503296, 3.785691738128662, 3.6986136436462402, 3.880495309829712, 3.550795316696167, 3.510852336883545, 3.7082650661468506, 3.4150662422180176, 3.7534728050231934, 3.4262521266937256, 3.6499269008636475, 3.502599000930786, 3.47599196434021, 3.702509880065918, 3.4912798404693604]
acc_das_pd_3=[sum(my_ifr_latency_pd_3[:i+1]) for i in range(len(my_ifr_latency_pd_3))]

lbs_ifr_latency_pd_6=\
    [4.82288932800293, 6.004542112350464, 7.228421926498413, 8.426594734191895, 9.662330150604248, 10.987264156341553, 8.317591667175293, 8.29173493385315, 8.340659618377686, 8.28456974029541, 8.2861328125, 8.374102592468262, 8.444041728973389, 8.526225805282593, 8.688915014266968, 8.90980577468872, 8.977296590805054, 8.77534008026123, 8.775488376617432, 8.808629035949707, 8.556262493133545, 8.319825172424316, 8.286192655563354, 8.35884404182434, 8.300535678863525, 8.48636531829834, 8.63257122039795, 8.638738632202148, 8.577716827392578, 8.831819295883179, 8.59686827659607, 8.507702589035034, 8.462507963180542, 8.621634244918823, 8.738494873046875, 8.373754024505615, 8.573064088821411, 8.481884717941284, 8.481877326965332, 8.441450595855713, 8.314927101135254, 8.377465724945068, 8.61679482460022, 8.477571249008179, 8.44596552848816, 8.526078939437866, 8.569299697875977, 8.6136953830719, 8.410275936126709, 8.33045744895935]
acc_lbs_pd_6=[sum(lbs_ifr_latency_pd_6[:i+1]) for i in range(len(lbs_ifr_latency_pd_6))]
my_ifr_latency_pd_6=\
    [3.5274691581726074, 4.667366027832031, 5.849313735961914, 6.318916320800781, 7.384417772293091, 8.371943950653076, 6.394896984100342, 6.5259480476379395, 6.447328329086304, 6.343261003494263, 6.370584011077881, 6.504204750061035, 6.419379472732544, 6.092080354690552, 6.353308916091919, 6.285302400588989, 6.336871385574341, 6.312177658081055, 6.554102182388306, 6.446082353591919, 6.24319863319397, 6.442186594009399, 6.306842565536499, 6.1844093799591064, 6.080895900726318, 6.211743593215942, 6.159480571746826, 5.971521377563477, 6.070272207260132, 6.052670955657959, 6.191030502319336, 6.179265737533569, 6.064193964004517, 6.545398235321045, 6.151921987533569, 6.451910734176636, 6.348672389984131, 6.484976291656494, 6.674726247787476, 6.395415782928467, 6.620257377624512, 6.625135183334351, 6.523777723312378, 6.446258783340454, 6.4146294593811035, 6.417489528656006, 6.364212274551392, 6.291529178619385, 6.31507134437561, 6.068163156509399]
acc_das_pd_6=[sum(my_ifr_latency_pd_6[:i+1]) for i in range(len(my_ifr_latency_pd_6))]
error1=[0.079,0.078,0.077]
error2=[0.076,0.075,0.075]
lbs=[3.95,1.68,1.48]
das=[2.57,1.25,1.1]
labels = ['1','3','6'] # 级别
index = np.arange(len(labels))
width = 0.20

fig, ax = plt.subplots(figsize=(5,4))
#ax.set_ylim(0,0.38)
rect1 = ax.bar(index - width/2, lbs,  width=width,color='NAVY', zorder=100,edgecolor = 'k',hatch = '///',label =r'LBS',yerr=error1,error_kw = {'ecolor’': '0.2', 'capsize' :3 })
rect3 = ax.bar(index+width/2, das,  width=width, color='white', zorder=100,edgecolor = 'r',hatch = '*',label =r'DAS',yerr=error2,error_kw = {'ecolor’': '0.2', 'capsize' :3 })
#ax.set_title('Quality gain',fontsize=14)
plt.tick_params(labelsize=15)
ax.set_xticks(ticks=index)
ax.set_xticklabels(labels,fontsize=15)
ax.set_ylabel('Average Processing Time (s)',fontsize=18)
ax.set_xlabel('Maximum Parallel Pipelines',fontsize=18)
ax.legend(loc='upper right',fontsize=20,ncol=1)
plt.grid(axis="y",linestyle='-.',zorder=0)
auto_text(rect1)
auto_text(rect3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

lbs_ifr_latency_pd_11= \
    [32.74787092208862, 32.39122247695923, 32.20626759529114, 32.31101083755493, 32.400633811950684, 32.25810194015503,
     32.42351269721985, 32.33753037452698, 32.375067949295044, 32.2475323677063, 32.42811417579651, 32.42941355705261,
     32.268364906311035, 32.23632764816284, 32.44314765930176, 32.37912321090698, 32.413286447525024, 32.48179388046265,
     32.40978646278381, 32.35944676399231, 32.32778811454773, 32.39806628227234, 32.259312868118286, 32.40842318534851,
     32.43529653549194, 32.4167218208313, 32.50453734397888, 32.44716763496399, 32.401368856430054, 32.30675482749939,
     32.34086513519287, 32.3001172542572, 32.5299870967865, 32.32141876220703, 32.42245173454285, 32.17583250999451,
     32.34539794921875, 32.52831840515137, 32.354655742645264, 32.37548851966858, 32.25824522972107, 32.454102754592896,
     32.3618049621582, 32.397032499313354, 32.26277542114258, 32.271448612213135, 32.697160720825195, 32.28840923309326,
     32.27875781059265, 32.36925721168518]
acc_lbs_pd_11=[sum(lbs_ifr_latency_pd_11[:i+1]) for i in range(len(lbs_ifr_latency_pd_11))]

my_ifr_latency_pd_11=ifr_latency=[29.780829906463623, 27.030491828918457, 26.694204092025757, 30.086950540542603, 30.091232776641846, 30.094482421875, 31.099443674087524, 31.237407684326172, 28.52865982055664, 29.427077054977417, 30.868412256240845, 28.682687520980835, 30.954453229904175, 29.567549228668213, 28.467912673950195, 28.468676328659058, 30.490301609039307, 29.366663932800293, 28.17548894882202, 30.029500722885132, 30.544949054718018, 30.620293140411377, 31.18926191329956, 29.737065076828003, 31.17815661430359, 29.658392906188965, 27.962013721466064, 28.57453989982605, 30.311699628829956, 29.78895854949951, 29.218281984329224, 30.24749183654785, 29.824774742126465, 27.66343593597412, 30.447425603866577, 30.384670734405518, 31.0046808719635, 28.302093267440796, 28.66732692718506, 28.96869921684265, 30.578164100646973, 29.75376605987549, 29.718197107315063, 31.123215675354004, 28.47199249267578, 29.008419513702393, 30.06100845336914, 29.765644073486328, 31.289026975631714, 29.237839698791504]
acc_das_pd_11=[sum(my_ifr_latency_pd_11[:i+1]) for i in range(len(my_ifr_latency_pd_11))]

lbs_ifr_latency_pd_33=\
    [34.601563692092896, 48.192505836486816, 62.07808184623718, 41.275452613830566, 41.12877440452576, 41.09971213340759, 41.0349235534668, 41.13252329826355, 40.95114350318909, 41.33625769615173, 41.40616273880005, 41.357590436935425, 41.713706731796265, 41.37950301170349, 41.54201912879944, 40.8137731552124, 40.78664493560791, 40.672441244125366, 41.34791088104248, 41.62864637374878, 41.20524287223816, 40.28082203865051, 40.041547775268555, 40.09900212287903, 40.69308090209961, 40.49249744415283, 40.807944536209106, 40.921494245529175, 41.20323157310486, 40.84851932525635, 40.17026114463806, 40.335036516189575, 40.19170928001404, 40.39684581756592, 40.34724044799805, 40.42596077919006, 40.598081827163696, 40.890379905700684, 40.90069603919983, 41.28103303909302, 40.810354471206665, 41.151020526885986, 41.1941351890564, 41.33648991584778, 41.21494174003601, 40.4493134021759, 40.18126392364502, 40.026094913482666, 40.50197672843933, 39.38094973564148]
acc_lbs_pd_33=[sum(lbs_ifr_latency_pd_33[:i+1]) for i in range(len(lbs_ifr_latency_pd_33))]

my_ifr_latency_pd_33= \
    [29.844707250595093, 43.7231171131134, 57.950096130371094, 41.26824331283569, 40.42070293426514, 39.380573987960815,
     39.7716646194458, 39.526029109954834, 39.1877064704895, 39.45990180969238, 39.64857053756714, 39.956032037734985,
     39.32725811004639, 39.045562505722046, 39.13056230545044, 39.39894962310791, 39.6806366443634, 39.47175359725952,
     39.484753131866455, 40.13023543357849, 40.78319215774536, 40.79171824455261, 40.481385707855225, 40.12850570678711,
     39.81274771690369, 39.4429612159729, 39.12810826301575, 39.064759731292725, 39.345398902893066, 39.180535554885864,
     39.2912483215332, 39.70903944969177, 39.53635501861572, 39.500420570373535, 39.21943712234497, 39.824015378952026,
     39.66541934013367, 39.33217096328735, 38.87159514427185, 39.06749653816223, 39.30023765563965, 39.66991901397705,
     39.72937083244324, 39.55108284950256, 39.440945863723755, 39.395514249801636, 39.672531843185425,
     39.88172173500061, 39.43157386779785, 38.26966643333435]
acc_das_pd_33=[sum(my_ifr_latency_pd_33[:i+1]) for i in range(len(my_ifr_latency_pd_33))]

lbs_ifr_latency_pd_66= \
    [35.45617890357971, 49.41926336288452, 62.82579302787781, 76.07109308242798, 90.00256395339966, 103.59415698051453,
     82.64442801475525, 81.85958361625671, 81.64868068695068, 81.8496527671814, 81.35271120071411, 81.16002035140991,
     80.7287290096283, 80.76396322250366, 80.87505578994751, 80.72329044342041, 80.69552612304688, 81.06320548057556,
     81.05856776237488, 81.23581981658936, 81.57488656044006, 81.71631097793579, 81.58663558959961, 80.99476599693298,
     80.94068121910095, 80.75018429756165, 80.74016761779785, 81.09310221672058, 80.97928619384766, 81.00840663909912,
     81.05057764053345, 81.10000038146973, 80.96640920639038, 80.83684611320496, 81.193603515625, 81.52595949172974,
     81.75658702850342, 82.2299063205719, 82.7421338558197, 82.79878330230713, 83.1504442691803, 83.57053303718567,
     83.49864339828491, 83.65850615501404, 83.43993091583252, 83.43350100517273, 83.00535154342651, 82.37229418754578,
     81.01099824905396, 79.1089596748352]
acc_lbs_pd_66=[sum(lbs_ifr_latency_pd_66[:i+1]) for i in range(len(lbs_ifr_latency_pd_66))]
my_ifr_latency_pd_66= \
    [29.973487377166748, 43.9328670501709, 59.07049107551575, 71.5870292186737, 84.84822368621826, 98.34390187263489,
     82.5528724193573, 81.58809494972229, 79.53950667381287, 79.71532440185547, 79.55407452583313, 79.55047035217285,
     79.10382103919983, 79.04307389259338, 78.85653519630432, 78.88201808929443, 78.86091256141663, 78.36765885353088,
     78.47584176063538, 79.11816906929016, 79.52351641654968, 79.53572154045105, 80.03549432754517, 80.45831608772278,
     80.41949009895325, 79.76797366142273, 79.39928317070007, 79.96809077262878, 79.47835636138916, 79.09559535980225,
     79.52021479606628, 79.4923963546753, 79.36889791488647, 78.73262739181519, 79.0824384689331, 79.39582467079163,
     78.86410808563232, 79.27962851524353, 79.39691042900085, 79.55155777931213, 79.30391764640808, 79.20012211799622,
     79.43804335594177, 79.01563096046448, 79.02144050598145, 78.62272691726685, 78.92113518714905, 78.57678198814392,
     72.36774396896362, 72.50326466560364]
acc_das_pd_66=[sum(my_ifr_latency_pd_66[:i+1]) for i in range(len(my_ifr_latency_pd_66))]
error1=[0.679,0.678,0.677]
error2=[0.676,0.475,0.475]
lbs=[32.38,14.02,14.01]
das=[29.65,12.43,12.38]
labels = ['1','3','6'] # 级别
index = np.arange(len(labels))
width = 0.20

fig, ax = plt.subplots(figsize=(5,4))
ax.set_ylim(0,35)
rect1 = ax.bar(index - width/2, lbs,  width=width,color='NAVY', zorder=100,edgecolor = 'k',hatch = '///',label =r'LBS',yerr=error1,error_kw = {'ecolor’': '0.2', 'capsize' :3 })
rect3 = ax.bar(index+width/2, das,  width=width, color='bisque', zorder=100,edgecolor = 'k',hatch = '*',label =r'DAS',yerr=error2,error_kw = {'ecolor’': '0.2', 'capsize' :3 })
#ax.set_title('Quality gain',fontsize=14)
plt.tick_params(labelsize=15)
ax.set_xticks(ticks=index)
ax.set_xticklabels(labels,fontsize=15)
ax.set_ylabel('Average Processing Time (s)',fontsize=18)
ax.set_xlabel('Maximum Parallel Pipelines',fontsize=18)
ax.legend(loc='upper right',fontsize=20,ncol=1)
plt.grid(axis="y",linestyle='-.',zorder=0)
auto_text1(rect1)
auto_text1(rect3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()