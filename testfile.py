import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.neighbors import KernelDensity

from vglm.process import TemperedStableProcess, TSLangevinModel
plt.style.use('ggplot')

# fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(ncols=2, nrows=2)
# fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(ncols=2, nrows=3)
fig, [ax5, ax6] = plt.subplots(ncols=2, nrows=1)
rng1 = np.random.default_rng(seed=2)
rng2 = np.random.default_rng(seed=2)
minj = -.5
cs = np.logspace(minj, minj, 5000)

fv = []
fvwr = []

for c in tqdm(cs):
	tslang = TSLangevinModel(x0=0., xd0=0., mu=0., sigmasq=1., kappa=0.5, delta=1., gamma=1., kv=1e-6, theta=-1., c=c, rng=rng1)
	tslang.generate(nobservations=50, res=False)
	# ax1.plot(tslang.observationtimes, tslang.observationvals, color='red', ls='--', lw=0., marker='.', ms=2., mec='black', mfc='black')
	# ax1.plot(tslang.observationtimes, tslang.underlyingvals, color='red', ls='--', lw=1.5)
	# ax1.plot(tslang.observationtimes, tslang.observationvals, ls='--', lw=1.5)
	# ax3.plot(tslang.observationtimes, tslang.observationgrad, ls='--', lw=1.5)
	fv.append(tslang.observationvals[-1])
for c in tqdm(cs):
	tslang = TSLangevinModel(x0=0., xd0=0., mu=0., sigmasq=1., kappa=0.5, delta=1., gamma=1., kv=1e-6, theta=-1., c=c, rng=rng2)
	tslang.generate(nobservations=50, res=True)
	# ax1.plot(tslang.observationtimes, tslang.observationvals, color='red', ls='--', lw=0., marker='.', ms=2., mec='black', mfc='black')
	# ax1.plot(tslang.observationtimes, tslang.underlyingvals, color='red', ls='--', lw=1.5)
	# ax2.plot(tslang.observationtimes, tslang.observationvals, ls='--', lw=1.5, label=str(c))
	# ax4.plot(tslang.observationtimes, tslang.observationgrad, ls='--', lw=1.5)
	fvwr.append(tslang.observationvals[-1])
# ax5.hist(fv, bins=100, density=True)
# ax6.hist(fvwr, bins=100, density=True)

fv = np.array(fv)[:, np.newaxis]
fvwr = np.array(fvwr)[:, np.newaxis]

kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(fv)
kdewr = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(fvwr)
axis = np.linspace(-15, 15, 10000)[:, np.newaxis]

ax5.plot(axis, (kde.score_samples(axis)), label='without res')
ax5.plot(axis, (kdewr.score_samples(axis)), label='with res')
ax6.plot(axis, np.exp(kde.score_samples(axis)))
ax6.plot(axis, np.exp(kdewr.score_samples(axis)))

fig.legend()
# fig.suptitle('min jump: e'+str(minj))
# ax1.set_title('Without residuals')
# ax2.set_title('With residuals')
# ax3.set_xlabel('time')
# ax4.set_xlabel('time')
# ax1.set_ylabel('x')
# ax3.set_ylabel('xdot')
plt.show()