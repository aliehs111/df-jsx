// src/components/QuickActions.jsx
import { Link } from 'react-router-dom'
import { PlusCircleIcon, FolderOpenIcon, ChatBubbleLeftEllipsisIcon } from '@heroicons/react/24/outline'
import newlogo500 from '../assets/newlogo500.png'

const logo = newlogo500


export default function QuickActions() {
  return (
    <div className="text-blue-800 rounded-lg bg-white p-6 shadow flex flex-col space-y-4">
      <h2 className="text-xl text-blue-800 font-semibold">Quick Actions</h2>
      <Link
        to="/upload"
        className="flex items-center space-x-2 rounded-md border text-blue-800 border-cyan-300 px-4 py-2 hover:bg-cyan-50"
      >
        <PlusCircleIcon className="h-6 w-6 text-blue-800" />
        <span>Upload New CSV</span>
      </Link>
      <Link
        to="/datasets"
        className="flex items-center space-x-2 rounded-md border border-cyan-300 text-blue-800 px-4 py-2 hover:bg-cyan-50"
      >
        <FolderOpenIcon className="h-6 w-6 text-blue-800" />
        <span>My Datasets</span>
      </Link>
      <Link
  to="/chat"
  className="mt-2 inline-flex items-center justify-center space-x-2 rounded-md bg-lime-500 px-4 py-2 text-white hover:bg-cyan-700"
>
  <ChatBubbleLeftEllipsisIcon className="h-6 w-6" />
  <span>Chat with DataBot...Coming Soon!</span>
  <img
    src={newlogo500}
    alt="DataBot logo"
    className="h-10 w-10 rounded-md"
  />
 
</Link>
    </div>
  )
}
