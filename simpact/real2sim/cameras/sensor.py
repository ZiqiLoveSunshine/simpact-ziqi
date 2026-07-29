from klampt.math import se3,so3,vectorops
from abc import ABC, abstractmethod
from dataclasses import dataclass
import dacite
from typing import Optional,Union,List,Dict

class Sensor(ABC):
    """A generic interface to a sensor.
    
    A sensor's connection is acquired upon initialization and released
    upon close() / destruction.
    
    The user will 1) pass any settings to the constructor, 2) repeatedly
    call update() to update the sensor's internal state, and 3) call
    value() to retrieve the sensor's value.

    The settings() method should return a dictionary of settings that
    can be passed to the constructor.  The constructor should accept
    these settings as keyword arguments.
    """
    def connected(self) -> bool:
        return True
    
    def settings(self) -> dict:
        return {}
    
    def rate(self) -> Optional[float]:
        """Returns the rate of the sensor in Hz.  Returns None to indicate
        an unknown rate."""
        return None

    @abstractmethod
    def update(self) -> bool:
        """Returns True if a new value is available, False otherwise."""
        pass

    @abstractmethod
    def channels(self) -> Union[int,List[str]]:
        """Retrieves the channels used by this sensor.  If the sensor has
        no channels and just returns an object, it should return 0.

        If the sensor returns a list of objects, it should return the number
        of objects in the list.

        If the sensor returns a dictionary of objects, it should return the
        keys of the dictionary.
        """
        pass

    @abstractmethod
    def value(self, id : Optional[Union[int,str]]=None) -> object:
        """Retrieves the value of the sensor from one or more channels.
        If id is None, returns all possible values as a list or dict.

        The resulting object is sensor-specific.
        """
        pass

    @abstractmethod
    def last_update(self) -> float:
        """Returns the time of the last retrieved value, in seconds since
        the epoch.
        """
        pass

    def close(self):
        pass

    def __del__(self):
        self.close()


class Multisensor(Sensor):
    """A sensor that aggregates multiple sensors as different channels."""
    def __init__(self, sensors : Dict[str,Sensor]):
        self._sensors = sensors
    
    def connected(self) -> bool:
        return all(s.connected() for s in self._sensors.values())
    
    def settings(self):
        return {k:s.settings() for k,s in self._sensors.items()}
    
    def rate(self) -> Optional[float]:
        return min(s.rate() for s in self._sensors.values())
    
    def update(self) -> bool:
        return any(s.update() for s in self._sensors.values())
    
    def channels(self) -> List[str]:
        res = []
        for k,s in self._sensors.items():
            ch = s.channels()
            if isinstance(ch,int):
                for i in range(ch):
                    res.append(k+'.'+str(i))
            else:
                res.extend([k + '.' + c for c in ch])
        return res
    
    def value(self, id : Optional[str]=None) -> object:
        if id is None:
            return {k:s.value() for k,s in self._sensors.items()}
        k,c = id.split('.')
        if c.isdigit():
            return self._sensors[k].value(int(c))
        return self._sensors[k].value(c)
    
    def last_update(self) -> float:
        return max(s.last_update() for s in self._sensors.values())
    
    def close(self):
        for s in self._sensors.values():
            s.close()
    
