using GTA.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace GetWorldInfo
{
    [Serializable]
    [XmlRoot("Screenshot")]
    [XmlInclude(typeof(Car))]
    [XmlInclude(typeof(Pedestrian))]
    public class Screenshot
    {
        [XmlArray("Cars")]
        [XmlArrayItem("Car")]
        public List<Car> Cars { get; set; }
        public Car MyCar { get; set; }

        [XmlArray("Peds")]
        [XmlArrayItem("Ped")]
        public List<Pedestrian> Peds { get; set; }
        public int Time { get; set; }
        public float LastFrameTime { get; set; }
        public int MyProperty { get; set; }
        public String Weather { get; set; }

        public int Steering { get; set; }
        public int Acc { get; set; }
        public int Brake { get; set; }
        public string DayTime { get; set; }

        
        public Vector3 Position { get; set; }
        
        public Vector3 Rotation { get; set; }


    }
}
