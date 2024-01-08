{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "source": [
    "import numpy as np\n",
    "samples = np.load('samples.npy')\n",
    "iq_fc32 = samples.astype(np.complex64)\n",
    "with open('samples.fc32', 'wb') as f: \n",
    "    f.write(iq_fc32.tobytes())\n",
    ""
   ],
   "outputs": [],
   "metadata": {}
  }
 ],
 "nbformat": 4,
 "nbformat_minor": 2,
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": 3
  }
 }
}