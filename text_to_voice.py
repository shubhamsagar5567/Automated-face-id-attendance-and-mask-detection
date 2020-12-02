import pyttsx3

flag = True #flag is set to true if mask is present
outText = "Mask is not detected, verfication failed"
if flag: outText = "Mask is detected verfication successful"

alexa = pyttsx3.init()
print("\nAlexa is Running....")

""" RATE"""
rate = alexa.getProperty('rate')   # getting details of current speaking rate
alexa.setProperty('rate', 135)     # setting up new voice rate
#voice rate is words per minute, ex: rate = 200 then 200 words per minute

"""VOLUME"""
volume = alexa.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
alexa.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

"""VOICE"""
voices = alexa.getProperty('voices')       #getting details of current voice
#alexa.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
alexa.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

alexa.say(outText)
print("\nAlexa is speaking....")
alexa.runAndWait()
alexa.stop()
print("\nAlexa is turned off")