#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to livox_ros_driver2__msg__CustomPoint
/// Livox costom pointcloud format.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CustomPoint {
    /// offset time relative to the base time
    pub offset_time: u32,

    /// X axis, unit:m
    pub x: f32,

    /// Y axis, unit:m
    pub y: f32,

    /// Z axis, unit:m
    pub z: f32,

    /// reflectivity, 0~255
    pub reflectivity: u8,

    /// livox tag
    pub tag: u8,

    /// laser number in lidar
    pub line: u8,

}



impl Default for CustomPoint {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::CustomPoint::default())
  }
}

impl rosidl_runtime_rs::Message for CustomPoint {
  type RmwMsg = super::msg::rmw::CustomPoint;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        offset_time: msg.offset_time,
        x: msg.x,
        y: msg.y,
        z: msg.z,
        reflectivity: msg.reflectivity,
        tag: msg.tag,
        line: msg.line,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      offset_time: msg.offset_time,
      x: msg.x,
      y: msg.y,
      z: msg.z,
      reflectivity: msg.reflectivity,
      tag: msg.tag,
      line: msg.line,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      offset_time: msg.offset_time,
      x: msg.x,
      y: msg.y,
      z: msg.z,
      reflectivity: msg.reflectivity,
      tag: msg.tag,
      line: msg.line,
    }
  }
}


// Corresponds to livox_ros_driver2__msg__CustomMsg
/// Livox publish pointcloud msg format.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CustomMsg {
    /// ROS standard message header
    pub header: std_msgs::msg::Header,

    /// The time of first point
    pub timebase: u64,

    /// Total number of pointclouds
    pub point_num: u32,

    /// Lidar device id number
    pub lidar_id: u8,

    /// Reserved use
    pub rsvd: [u8; 3],

    /// Pointcloud data
    pub points: Vec<super::msg::CustomPoint>,

}



impl Default for CustomMsg {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::CustomMsg::default())
  }
}

impl rosidl_runtime_rs::Message for CustomMsg {
  type RmwMsg = super::msg::rmw::CustomMsg;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        timebase: msg.timebase,
        point_num: msg.point_num,
        lidar_id: msg.lidar_id,
        rsvd: msg.rsvd,
        points: msg.points
          .into_iter()
          .map(|elem| super::msg::CustomPoint::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
      timebase: msg.timebase,
      point_num: msg.point_num,
      lidar_id: msg.lidar_id,
        rsvd: msg.rsvd,
        points: msg.points
          .iter()
          .map(|elem| super::msg::CustomPoint::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      timebase: msg.timebase,
      point_num: msg.point_num,
      lidar_id: msg.lidar_id,
      rsvd: msg.rsvd,
      points: msg.points
          .into_iter()
          .map(super::msg::CustomPoint::from_rmw_message)
          .collect(),
    }
  }
}


