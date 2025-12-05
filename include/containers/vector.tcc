// //! Add an element pointer
// template <class T>
// bool mpm::Vector<T>::add(const std::shared_ptr<T>& ptr, bool check_duplicates) {
//   bool insertion_status = false;
//   if (check_duplicates) {
//     // Check if it is found in the Vector
//     auto itr = std::find_if(this->cbegin(), this->cend(),
//                             [ptr](std::shared_ptr<T> const& element) {
//                               return element->id() == ptr->id();
//                             });

//     if (itr == this->cend()) {
//       elements_.push_back(ptr);
//       insertion_status = true;
//     }
//   } else {
//     elements_.push_back(ptr);
//     insertion_status = true;
//   }
//   return insertion_status;
// }

// //! Remove a pointer
// template <class T>
// bool mpm::Vector<T>::remove(const std::shared_ptr<T>& ptr) {
//   auto size = elements_.size();
//   // Check if it is found in the Vector
//   elements_.erase(std::remove(elements_.begin(), elements_.end(), ptr),
//                   elements_.end());
//   return !(size == elements_.size());
// }

// //! Iterate over elements in the Vector
// template <class T>
// template <class Tunaryfn>
// Tunaryfn mpm::Vector<T>::for_each(Tunaryfn fn) {
//   return std::for_each(elements_.begin(), elements_.end(), fn);
// }




//! Add an element pointer
template <class T>
bool mpm::Vector<T>::add(const std::shared_ptr<T>& ptr, bool check_duplicates) {
  if (check_duplicates) {
    // O(N)の線形探索の代わりに、O(1)のハッシュ検索を使う
    const auto id = ptr->id();
    if (existing_ids_.find(id) == existing_ids_.end()) {
      // 見つからなかった場合のみ要素を追加
      elements_.push_back(ptr);
      existing_ids_.insert(id); // IDをセットに追加
      return true;
    }
    return false; // 重複していたので追加しない
  } else {
    elements_.push_back(ptr);
    // こちらの場合も、もし重複管理が必要ならIDをセットに追加すべきか検討
    existing_ids_.insert(ptr->id()); 
    return true;
  }
}

//! Remove a pointer
template <class T>
bool mpm::Vector<T>::remove(const std::shared_ptr<T>& ptr) {
  const auto id_to_remove = ptr->id();

  // 1. elements_から要素を削除
  auto itr = std::remove(elements_.begin(), elements_.end(), ptr);
  if (itr != elements_.end()) {
    elements_.erase(itr, elements_.end());

    // 2. existing_ids_からIDを削除
    existing_ids_.erase(id_to_remove); // ★★★ 同期のために必ず削除

    return true;
  }
  return false;
}